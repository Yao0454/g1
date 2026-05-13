"""
core/voice.py — 语音模块：唤醒词 → VAD 录音 → ASR → Chat → 让 G1 喇叭念。

封装成 Voice 类，外面只需要：
    bridge = Bridge()
    chat   = Chat()
    voice  = Voice(bridge, chat=chat)
    voice.start()           # 后台线程跑唤醒-对话循环；主线程继续干别的（比如 vision.run()）
    voice.say("我在")       # 也可以主动让机器人说一句
    voice.stop()            # 关线程

子系统：
    · ASR  : sherpa-onnx SenseVoice （CUDA）
    · KWS  : sherpa-onnx zipformer wenetspeech 拼音建模 （CPU）
    · VAD  : silero_vad （CPU，自动断句）
    · TTS  : 默认走 bridge.send_tts → C++ → G1 自带喇叭；可选 melo-tts 本地播 EarPods
    · MicStream : 一条常开 sounddevice 输入流（避免每句新开流导致句首被吃掉）

回声坑（G1 喇叭说的话被 EarPods 麦克风拾到）的处理：
    · 唤醒确认用本地 beep()（80ms 短"嘀"），不走 TTS，响完 mic.drain() 一次
    · TTS 走 G1 喇叭是异步的（UDP 发完就返回），按字数估个时长 sleep + drain，期间不收麦
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

from .bridge import Bridge

# ── 路径 ─────────────────────────────────────────────────────────────────────
HOME = Path.home()
ASR_DIR = HOME / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
TTS_DIR = HOME / "vits-melo-tts-zh_en"
KWS_DIR = HOME / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
VAD_MODEL = HOME / "silero_vad.onnx"
# keywords.txt 跟 g1.py / talk.py 同级（package 外面）
KEYWORDS_FILE = Path(__file__).resolve().parent.parent / "keywords.txt"
DEFAULT_WAKE_WORDS = ["你好机器人", "你好宇树"]

SAMPLE_RATE = 16000
ASR_PROVIDER = "cuda"
TTS_PROVIDER = "cpu"    # melo-tts onnx 在 CUDA EP 上长句会 Reshape 崩
KWS_PROVIDER = "cpu"


# ── PulseAudio 默认录音源自动纠正（USB 麦拔插后会漂走）─────────────────────
def ensure_mic_default(verbose: bool = True) -> None:
    if not shutil.which("pactl"):
        return
    try:
        cur = subprocess.run(["pactl", "get-default-source"],
                             capture_output=True, text=True).stdout.strip()
        if not cur:
            info = subprocess.run(["pactl", "info"], capture_output=True, text=True).stdout
            for ln in info.splitlines():
                if ln.startswith("Default Source:"):
                    cur = ln.split(":", 1)[1].strip()
        srcs = subprocess.run(["pactl", "list", "short", "sources"],
                              capture_output=True, text=True).stdout
        names = [ln.split("\t")[1] for ln in srcs.splitlines() if "\t" in ln]
        real = [n for n in names if not n.endswith(".monitor") and "platform-sound" not in n]
        usb = [n for n in real if "usb" in n.lower()] or real
        if not usb:
            if verbose:
                print("[MIC] ⚠️  PulseAudio 没看到 USB 麦克风，确认插好了？", file=sys.stderr)
            return
        tgt = usb[0]
        if cur == tgt:
            if verbose:
                print(f"[MIC] 录音源 = {tgt}")
            return
        subprocess.run(["pactl", "set-default-source", tgt], check=False)
        out = tgt.replace("alsa_input.", "alsa_output.")
        sinks = subprocess.run(["pactl", "list", "short", "sinks"],
                               capture_output=True, text=True).stdout
        sink_names = [ln.split("\t")[1] for ln in sinks.splitlines() if "\t" in ln]
        if out in sink_names:
            subprocess.run(["pactl", "set-default-sink", out], check=False)
        if verbose:
            print(f"[MIC] 默认录音源 {cur or '(空)'} → {tgt}（已切）")
    except Exception as e:
        if verbose:
            print(f"[MIC] 检查录音源出错（忽略）：{e}", file=sys.stderr)


# ── sounddevice 懒加载 ──────────────────────────────────────────────────────
def _import_sd():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("缺 sounddevice：pip install sounddevice（系统装 libportaudio2）")


def beep(freq: float = 880.0, dur: float = 0.12, vol: float = 0.25,
         sample_rate: int = SAMPLE_RATE) -> None:
    """唤醒确认音，比 VAD min_speech 短，响完 drain 一下不会回灌。"""
    try:
        sd = _import_sd()
        t = np.arange(int(dur * sample_rate)) / sample_rate
        wav = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        fade = max(1, int(0.005 * sample_rate))
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
        sd.play(wav, samplerate=sample_rate)
        sd.wait()
    except Exception:
        pass


# ── 常开麦克风流 ────────────────────────────────────────────────────────────
class MicStream:
    """0.1s 一块地 read()。常开 → 不会吃句首。"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        sd = _import_sd()
        self.sample_rate = sample_rate
        self.block = int(0.1 * sample_rate)
        self._stream = sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate)
        self._stream.start()
        time.sleep(0.25)
        self.drain()

    def read(self) -> np.ndarray:
        data, _ = self._stream.read(self.block)
        return data.reshape(-1).copy()

    def drain(self) -> None:
        try:
            n = self._stream.read_available
            while n:
                self._stream.read(n)
                n = self._stream.read_available
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


# ── ASR ─────────────────────────────────────────────────────────────────────
class _ASR:
    def __init__(self, model_dir: Path = ASR_DIR, provider: str = ASR_PROVIDER):
        model = model_dir / "model.int8.onnx"
        tokens = model_dir / "tokens.txt"
        if not model.exists():
            raise FileNotFoundError(f"找不到 ASR 模型 {model}")
        print(f"[ASR] 加载 SenseVoice ({provider}) …", flush=True)
        t0 = time.time()
        self.r = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=2,
            use_itn=True, language="zh", provider=provider, debug=False)
        print(f"[ASR] 就绪 {time.time() - t0:.1f}s")

    def transcribe(self, samples: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        s = self.r.create_stream()
        s.accept_waveform(sr, samples)
        self.r.decode_stream(s)
        return s.result.text.strip()


# ── 本地 TTS（可选，调试用；默认走 G1 喇叭）─────────────────────────────────
class _LocalTTS:
    def __init__(self, model_dir: Path = TTS_DIR, provider: str = TTS_PROVIDER):
        model = model_dir / "model.onnx"
        if not model.exists():
            raise FileNotFoundError(f"找不到 TTS 模型 {model}")
        rule_fsts = ",".join(str(model_dir / f) for f in ("date.fst", "number.fst", "phone.fst")
                             if (model_dir / f).exists())
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model),
                    lexicon=str(model_dir / "lexicon.txt"),
                    tokens=str(model_dir / "tokens.txt")),
                provider=provider, num_threads=4, debug=False),
            rule_fsts=rule_fsts, max_num_sentences=1)
        if not cfg.validate():
            raise RuntimeError("本地 TTS 配置校验失败")
        print(f"[TTS] 加载 melo-tts ({provider}) …", flush=True)
        t0 = time.time()
        self.tts = sherpa_onnx.OfflineTts(cfg)
        print(f"[TTS] 就绪 {time.time() - t0:.1f}s")

    def say(self, text: str) -> None:
        if not text.strip():
            return
        gc = sherpa_onnx.GenerationConfig()
        gc.speed = 1.0
        a = self.tts.generate(text, gc)
        sd = _import_sd()
        sd.play(np.asarray(a.samples, dtype=np.float32), samplerate=a.sample_rate)
        sd.wait()


# ── 唤醒词 ──────────────────────────────────────────────────────────────────
def _ensure_keywords_file(phrases, kws_dir: Path = KWS_DIR,
                          out_path: Path = KEYWORDS_FILE) -> Path:
    from sherpa_onnx import text2token
    tokens_path = kws_dir / "tokens.txt"
    encoded = text2token([[p] for p in phrases], tokens=str(tokens_path), tokens_type="ppinyin")
    lines = [" ".join(tok) + f" @{p}" for p, tok in zip(phrases, encoded)]
    content = "\n".join(lines) + "\n"
    if not (out_path.exists() and out_path.read_text(encoding="utf-8") == content):
        out_path.write_text(content, encoding="utf-8")
        print(f"[KWS] 唤醒词写入 {out_path}")
        for ln in lines:
            print(f"      {ln}")
    return out_path


class _Wake:
    def __init__(self, phrases, kws_dir: Path = KWS_DIR,
                 provider: str = KWS_PROVIDER, threshold: float = 0.25, score: float = 1.0):
        enc = next(kws_dir.glob("encoder-*chunk-16-left-64.onnx"))
        dec = next(kws_dir.glob("decoder-*chunk-16-left-64.onnx"))
        joi = next(kws_dir.glob("joiner-*chunk-16-left-64.onnx"))
        kw = _ensure_keywords_file(phrases, kws_dir)
        print(f"[KWS] 加载 zipformer-kws ({provider}) …", flush=True)
        t0 = time.time()
        self.s = sherpa_onnx.KeywordSpotter(
            tokens=str(kws_dir / "tokens.txt"), encoder=str(enc), decoder=str(dec), joiner=str(joi),
            keywords_file=str(kw), num_threads=2,
            keywords_threshold=threshold, keywords_score=score, provider=provider)
        print(f"[KWS] 就绪 {time.time() - t0:.1f}s  唤醒词: {', '.join(phrases)}")

    def wait(self, mic: MicStream, stop: threading.Event, debug: bool = False) -> str | None:
        st = self.s.create_stream()
        last = time.time()
        peak = 0.0
        while not stop.is_set():
            arr = mic.read()
            peak = max(peak, float(np.abs(arr).max()))
            st.accept_waveform(mic.sample_rate, arr)
            while self.s.is_ready(st):
                self.s.decode_stream(st)
                r = self.s.get_result(st)
                if r:
                    self.s.reset_stream(st)
                    return r
            if debug and time.time() - last >= 3.0:
                print(f"  [KWS] 监听… 近3s峰值 {peak:.4f}", flush=True)
                last, peak = time.time(), 0.0
        return None


# ── VAD ─────────────────────────────────────────────────────────────────────
def _make_vad(min_silence: float = 0.6):
    if not VAD_MODEL.exists():
        raise FileNotFoundError(f"找不到 VAD 模型 {VAD_MODEL}")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD_MODEL)
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = 0.1
    cfg.silero_vad.threshold = 0.45
    cfg.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


def _vad_capture(vad, mic: MicStream, stop: threading.Event, *,
                 max_seconds: float = 12.0, start_timeout: float = 6.0,
                 debug: bool = False) -> np.ndarray:
    vad.reset()
    spoke = False
    t_start = time.time()
    t_speech = None
    peak = 0.0
    last = t_start
    print("[LISTEN] 请说话…", flush=True)
    while not stop.is_set():
        arr = mic.read()
        peak = max(peak, float(np.abs(arr).max()))
        vad.accept_waveform(arr)
        if vad.is_speech_detected() and not spoke:
            spoke = True
            t_speech = time.time()
            if debug:
                print("  [LISTEN] 检测到说话…", flush=True)
        if not vad.empty():
            seg = np.asarray(vad.front.samples, dtype=np.float32)
            vad.pop()
            print(f"[LISTEN] 录到 {len(seg) / mic.sample_rate:.1f}s "
                  f"(peak {float(np.abs(seg).max()):.3f})")
            return seg
        now = time.time()
        if debug and now - last >= 2.0:
            tag = "录音中" if spoke else "等待中"
            warn = "  ←峰值<0.01，麦克风没拾到音！" if peak < 0.01 else ""
            print(f"  [LISTEN] {tag}… 近2s峰值 {peak:.4f}{warn}", flush=True)
            last, peak = now, 0.0
        if not spoke:
            if now - t_start > start_timeout:
                print(f"[LISTEN] {start_timeout:.0f}s 没人说话，跳过")
                return np.zeros(0, dtype=np.float32)
        elif now - t_speech > max_seconds:
            print(f"[LISTEN] 说超过 {max_seconds:.0f}s，截断")
            vad.flush()
            if not vad.empty():
                seg = np.asarray(vad.front.samples, dtype=np.float32)
                vad.pop()
                return seg
            return np.zeros(0, dtype=np.float32)
    return np.zeros(0, dtype=np.float32)


def _prep_for_asr(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = float(np.abs(samples).max())
    if 1e-4 < peak < 0.95:
        samples = samples * (0.7 / peak)
    pad = np.zeros(int(0.1 * sr), dtype=np.float32)
    return np.concatenate([pad, samples.astype(np.float32), pad])


# ── Voice 主类 ──────────────────────────────────────────────────────────────
class Voice:
    def __init__(self, bridge: Bridge, *,
                 chat=None,
                 wake_words: list[str] | None = None,
                 wake_threshold: float = 0.25,
                 listen_seconds: float = 12.0,
                 local_tts: bool = False,
                 verbose: bool = False):
        """
        bridge: 走 G1 喇叭的通道（bridge.send_tts(text)）
        chat:   有的话每次识别完会跑 chat.reply(text) → 把回复说出来；没有则只识别打印
        local_tts: True 用本地 melo-tts 经 EarPods 播；False（默认）走 G1 喇叭
        """
        self.bridge = bridge
        self.chat = chat
        self.wake_words = wake_words or list(DEFAULT_WAKE_WORDS)
        self.wake_threshold = wake_threshold
        self.listen_seconds = listen_seconds
        self.local_tts_on = local_tts
        self.verbose = verbose

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._asr: _ASR | None = None
        self._wake: _Wake | None = None
        self._vad = None
        self._mic: MicStream | None = None
        self._local_tts: _LocalTTS | None = None

    # ── 启动/停止 ───────────────────────────────────────────────────────────
    def _load(self) -> None:
        ensure_mic_default()
        try:
            sd = _import_sd()
            dev = sd.query_devices(sd.default.device[0])
            print(f"[MIC] 输入设备: {dev['name']}  ({dev['default_samplerate']:.0f}Hz)")
        except Exception as e:
            print(f"[MIC] 列设备失败（忽略）：{e}", file=sys.stderr)

        self._asr = _ASR()
        self._wake = _Wake(self.wake_words, threshold=self.wake_threshold)
        self._vad = _make_vad()
        self._mic = MicStream()
        if self.local_tts_on:
            self._local_tts = _LocalTTS()

    def start(self) -> None:
        """开后台线程跑唤醒-对话循环。"""
        if self._thread and self._thread.is_alive():
            return
        self._load()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Voice")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._mic:
            self._mic.close()

    # ── 高级接口 ────────────────────────────────────────────────────────────
    def say(self, text: str) -> float:
        """让机器人说一句。返回估计的"还得忙几秒"（G1 喇叭异步），调用方按需 sleep。
        本地 TTS 是阻塞的，返 0。"""
        text = text.strip()
        if not text:
            return 0.0
        if self._local_tts is not None:
            t0 = time.time()
            self._local_tts.say(text)
            if self.verbose:
                print(f"        (本地 TTS {time.time() - t0:.2f}s)")
            return 0.0
        self.bridge.send_tts(text)
        est = 1.3 + 0.22 * len(text)
        if self.verbose:
            print(f"        (→ G1 喇叭，约 {est:.1f}s)")
        return est

    def _settle(self, busy: float) -> None:
        """机器人念完之前不收麦：sleep + drain 麦克风缓冲。"""
        if busy and busy > 0:
            time.sleep(busy)
        if self._mic:
            self._mic.drain()

    # ── 调试用：纯听写模式（VAD 断句 + ASR 打印，不走唤醒/不对话/不出声）────
    def hear(self) -> None:
        """阻塞：一直 VAD 断句 → ASR 打印识别结果。Ctrl-C / stop() 退。
        需要先 start() 不然没加载模型；不过为了能独立用，这里允许懒加载。"""
        if self._asr is None or self._mic is None:
            ensure_mic_default()
            self._asr = _ASR()
            self._vad = _make_vad()
            self._mic = MicStream()
        print("=== Voice.hear()：随便说话，断句即识别；Ctrl-C 退 ===")
        last = time.time()
        peak = 0.0
        try:
            while not self._stop.is_set():
                arr = self._mic.read()
                peak = max(peak, float(np.abs(arr).max()))
                self._vad.accept_waveform(arr)
                if not self._vad.empty():
                    seg = np.asarray(self._vad.front.samples, dtype=np.float32)
                    self._vad.pop()
                    print(f"[LISTEN] 录到 {len(seg) / self._mic.sample_rate:.1f}s "
                          f"(peak {float(np.abs(seg).max()):.3f})")
                    if seg.size:
                        t0 = time.time()
                        text = self._asr.transcribe(_prep_for_asr(seg))
                        print(f"[识别] {text!r}   (ASR {time.time() - t0:.2f}s)")
                if time.time() - last >= 3.0:
                    warn = "  ←峰值<0.01，麦克风没拾到音！" if peak < 0.01 else ""
                    print(f"  [LISTEN] 监听… 近3s峰值 {peak:.4f}{warn}", flush=True)
                    last, peak = time.time(), 0.0
        except KeyboardInterrupt:
            print("\n[VOICE] Ctrl-C 退出")
        finally:
            if self._mic:
                self._mic.close()

    # ── 主循环（后台线程跑）─────────────────────────────────────────────────
    def _loop(self) -> None:
        assert self._wake and self._asr and self._mic
        self._settle(self.say("我准备好了"))
        print(f"=== Voice 起飞，唤醒词 {self.wake_words}；voice.stop() 退出 ===")
        try:
            while not self._stop.is_set():
                print(f"\n[KWS] 等待唤醒词 …", flush=True)
                hit = self._wake.wait(self._mic, self._stop, debug=self.verbose)
                if hit is None:
                    break
                print(f"[KWS] 唤醒：{hit}")
                beep()
                self._mic.drain()
                samples = _vad_capture(self._vad, self._mic, self._stop,
                                       max_seconds=self.listen_seconds,
                                       debug=self.verbose)
                if samples.size == 0:
                    continue
                t0 = time.time()
                text = self._asr.transcribe(_prep_for_asr(samples))
                print(f"[识别] {text!r}   (ASR {time.time() - t0:.2f}s)")
                if not text:
                    continue
                if self.chat is None:
                    continue
                t0 = time.time()
                reply = self.chat.reply(text)
                print(f"[回复] {reply}   (LLM {time.time() - t0:.2f}s)")
                self._settle(self.say(reply))
        except Exception as e:
            print(f"[VOICE] 循环异常退出：{e}", file=sys.stderr)
