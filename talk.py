"""
talk.py — 语音助手：唤醒词 → 录音(VAD) → ASR → llm.py → TTS 出声 [+ UDP 发文本给 C++ talk]

三段都用 sherpa-onnx（已编译好，在 vision env 跑）：
  ASR  : SenseVoice 离线识别（CUDA）        ~/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
  唤醒 : KWS zipformer wenetspeech（拼音建模，CPU）  ~/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/
  VAD  : silero_vad（CPU，唤醒后自动判断说完没）       ~/silero_vad.onnx
  TTS  : vits-melo-tts-zh_en（中英双语，CPU —— 这模型的 onnx 在 CUDA EP 上长句会 Reshape 崩，用 CPU，RTF≈0.7）  ~/vits-melo-tts-zh_en/

用法：
  python talk.py some.wav [--chat]      # 转写一个 wav（--chat 再跑一轮对话）
  python talk.py --hear                 # 调试：一直说话，它就断句 + 打印识别结果（不唤醒/不对话/不出声）
  python talk.py --mic [--chat]         # 手动：回车开始/再回车停止 录一段 → 转写[ → 对话]
  python talk.py --loop                 # 连续语音对话：回车录一句 → ASR → llm → TTS（Ctrl-C 退）
  python talk.py --wake                 # 全程免手：等"你好机器人" → VAD 录你说的话 → ASR → llm → 本地 TTS（EarPods 出声）
  python talk.py --wake --g1            # 同上，但回复交给 G1 自带喇叭念（经 ~/unitree_sdk2/build/bin/talk 节点）
  常用开关：--seconds N（手动模式改固定录 N 秒）  --wake-words "你好机器人,小宇小宇"  --wake-threshold 0.2
            --listen-seconds 12   --debug（打麦克风峰值/VAD起停）   --save-audio（每段录音存 /tmp/talk_heard_N.wav）

出声有两条路（独立，可单用可都用）：
  · 本地 TTS  : melo-tts → sounddevice 放到 EarPods（默认开；--no-tts / --g1 关掉，启动也更快）
  · G1 喇叭   : 回复文本 UDP 发到 127.0.0.1:9872 → C++ `talk` 节点调 G1 AudioClient.TtsMaker()
                先在另一终端：  ~/unitree_sdk2/build/bin/talk <网口，如 eth0>
                python 这边 --udp-port 0 可关掉 UDP 发送

依赖（都已装好）：sherpa_onnx（conda activate vision 自动带 PYTHONPATH）、sounddevice + libportaudio2、
                  pypinyin + sentencepiece（生成唤醒词拼音用）。
默认 PulseAudio 录/放设备都指向 USB EarPods；talk.py 启动时会 ensure_mic_default() 自动纠正。
"""

import argparse
import queue
import socket
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

# ── 路径 ──────────────────────────────────────────────────────────────────────
HOME = Path.home()
ASR_DIR = HOME / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
TTS_DIR = HOME / "vits-melo-tts-zh_en"
KWS_DIR = HOME / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
VAD_MODEL = HOME / "silero_vad.onnx"
KEYWORDS_FILE = Path(__file__).resolve().parent / "keywords.txt"   # 自动生成
DEFAULT_WAKE_WORDS = ["你好机器人", "你好宇树"]

SAMPLE_RATE = 16000
ASR_PROVIDER = "cuda"   # SenseVoice 在 CUDA 上没问题
TTS_PROVIDER = "cpu"    # melo-tts onnx 在 CUDA EP 上长句 Reshape 崩；CPU 够用
KWS_PROVIDER = "cpu"    # 模型才 3.3M，CPU 即可


# ── WAV ───────────────────────────────────────────────────────────────────────
def read_wave(path: str):
    """读 16-bit PCM wav，返回 (float32 单声道样本 in [-1,1], sample_rate)。"""
    with wave.open(path) as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: 只支持 16-bit PCM wav")
        num_channels = f.getnchannels()
        sample_rate = f.getframerate()
        raw = f.readframes(f.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels)[:, 0].copy()
    return samples, sample_rate


def write_wave(path: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """调试用：把 float32 样本存成 16-bit wav。"""
    pcm = np.clip(np.asarray(samples) * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


# ── ASR（SenseVoice）──────────────────────────────────────────────────────────
class ASR:
    def __init__(self, model_dir: Path = ASR_DIR, provider: str = ASR_PROVIDER,
                 num_threads: int = 2, language: str = "zh", use_itn: bool = True):
        model = model_dir / "model.int8.onnx"
        tokens = model_dir / "tokens.txt"
        if not model.exists():
            raise FileNotFoundError(f"找不到 ASR 模型 {model}（见文件顶部注释下载 SenseVoice）")
        print(f"[ASR] 加载 SenseVoice ({provider}) ...", flush=True)
        t0 = time.time()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=num_threads,
            use_itn=use_itn, language=language, provider=provider, debug=False)
        print(f"[ASR] 就绪 {time.time() - t0:.1f}s")

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_file(self, wav_path: str) -> str:
        samples, sr = read_wave(wav_path)
        return self.transcribe_samples(samples, sr)


# ── TTS（melo-tts 中英双语）────────────────────────────────────────────────────
class TTS:
    def __init__(self, model_dir: Path = TTS_DIR, provider: str = TTS_PROVIDER,
                 num_threads: int = 4, speed: float = 1.0):
        model = model_dir / "model.onnx"
        if not model.exists():
            raise FileNotFoundError(f"找不到 TTS 模型 {model}（见文件顶部注释下载 vits-melo-tts-zh_en）")
        # 这模型自带 date/number/phone 规整 fst（念数字、日期更顺）
        rule_fsts = ",".join(str(model_dir / f) for f in ("date.fst", "number.fst", "phone.fst")
                             if (model_dir / f).exists())
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model),
                    lexicon=str(model_dir / "lexicon.txt"),
                    tokens=str(model_dir / "tokens.txt")),
                provider=provider, num_threads=num_threads, debug=False),
            rule_fsts=rule_fsts, max_num_sentences=1)
        if not cfg.validate():
            raise RuntimeError("TTS 配置校验失败")
        print(f"[TTS] 加载 melo-tts ({provider}) ...", flush=True)
        t0 = time.time()
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.speed = speed
        print(f"[TTS] 就绪 {time.time() - t0:.1f}s  sr={self.tts.sample_rate}")

    def synth(self, text: str):
        gc = sherpa_onnx.GenerationConfig()
        gc.speed = self.speed
        a = self.tts.generate(text, gc)
        return np.asarray(a.samples, dtype=np.float32), a.sample_rate

    def say(self, text: str) -> None:
        if not text.strip():
            return
        samples, sr = self.synth(text)
        play_audio(samples, sr)


# ── 唤醒词（KWS）──────────────────────────────────────────────────────────────
def ensure_keywords_file(phrases, kws_dir: Path = KWS_DIR, out_path: Path = KEYWORDS_FILE):
    """把中文唤醒词转成模型要的拼音 token 行写到 out_path（已存在且词一致就不重写）。"""
    from sherpa_onnx import text2token
    tokens_path = kws_dir / "tokens.txt"
    encoded = text2token([[p] for p in phrases], tokens=str(tokens_path), tokens_type="ppinyin")
    lines = [" ".join(tok) + f" @{p}" for p, tok in zip(phrases, encoded)]
    new_content = "\n".join(lines) + "\n"
    if out_path.exists() and out_path.read_text(encoding="utf-8") == new_content:
        return out_path
    out_path.write_text(new_content, encoding="utf-8")
    print(f"[KWS] 唤醒词文件已生成 {out_path}:")
    for ln in lines:
        print(f"      {ln}")
    return out_path


class WakeWord:
    def __init__(self, phrases, kws_dir: Path = KWS_DIR, provider: str = KWS_PROVIDER,
                 num_threads: int = 2, threshold: float = 0.25, score: float = 1.0):
        enc = next(kws_dir.glob("encoder-*chunk-16-left-64.onnx"))
        dec = next(kws_dir.glob("decoder-*chunk-16-left-64.onnx"))
        joi = next(kws_dir.glob("joiner-*chunk-16-left-64.onnx"))
        kw_file = ensure_keywords_file(phrases, kws_dir)
        print(f"[KWS] 加载 zipformer-kws ({provider}) ...", flush=True)
        t0 = time.time()
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(kws_dir / "tokens.txt"), encoder=str(enc), decoder=str(dec), joiner=str(joi),
            keywords_file=str(kw_file), num_threads=num_threads,
            keywords_threshold=threshold, keywords_score=score, provider=provider)
        print(f"[KWS] 就绪 {time.time() - t0:.1f}s  唤醒词: {', '.join(phrases)}")

    def wait(self, mic: "MicStream", debug: bool = False) -> str:
        """从常开麦克风流读，直到检测到任一唤醒词，返回命中的词。"""
        stream = self.spotter.create_stream()
        last_report = time.time()
        peak_win = 0.0
        while True:
            arr = mic.read()
            peak_win = max(peak_win, float(np.abs(arr).max()))
            stream.accept_waveform(mic.sample_rate, arr)
            while self.spotter.is_ready(stream):
                self.spotter.decode_stream(stream)
                r = self.spotter.get_result(stream)
                if r:
                    self.spotter.reset_stream(stream)
                    return r
            if debug and time.time() - last_report >= 3.0:
                print(f"  [KWS] 监听中… 近3s峰值 {peak_win:.4f}", flush=True)
                last_report, peak_win = time.time(), 0.0


# ── 常开麦克风流 ──────────────────────────────────────────────────────────────
class MicStream:
    """一条从头开到尾的麦克风输入流，0.1s 一块地 read()。
    好处：不用每句话新开流（开流头 0.2~0.3s 是 ALSA 启动毛刺/空白，会把句首吃掉 → 识别变残）。"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        sd = _import_sd()
        self.sample_rate = sample_rate
        self.block = int(0.1 * sample_rate)
        self._stream = sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate)
        self._stream.start()
        time.sleep(0.25)        # 等 ALSA 稳定
        self.drain()            # 丢掉启动那点毛刺

    def read(self) -> np.ndarray:
        data, _ = self._stream.read(self.block)
        return data.reshape(-1).copy()

    def drain(self) -> None:
        """丢掉缓冲里积压的样本（比如刚播完 TTS，不想把自己的声音喂进 VAD/KWS）。"""
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


# ── VAD：自动录到你说完 ───────────────────────────────────────────────────────
def make_vad(vad_model: Path = VAD_MODEL, sample_rate: int = SAMPLE_RATE,
             min_silence: float = 0.6):
    if not vad_model.exists():
        raise FileNotFoundError(
            f"找不到 VAD 模型 {vad_model}\n下载：curl -L -o {vad_model} "
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(vad_model)
    cfg.silero_vad.min_silence_duration = min_silence   # 静音这么久算一句说完
    cfg.silero_vad.min_speech_duration = 0.1            # 0.1s 起就算说话（短词如"你好"也不漏）
    cfg.silero_vad.threshold = 0.45                     # 略低，句首/小声更容易起判
    cfg.sample_rate = sample_rate
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


def vad_capture(vad, mic: "MicStream", *, max_seconds: float = 12.0,
                start_timeout: float = 6.0, debug: bool = False) -> np.ndarray:
    """从常开麦克风流喂 VAD：等开口 → 录到停 → 返回这一段（float32 单声道）。
    没开口超过 start_timeout 返回空；开口后说超过 max_seconds 截断返回。"""
    vad.reset()
    spoke = False
    t_start = time.time()
    t_speech = None
    peak_win = 0.0
    last_report = t_start
    print("[LISTEN] 请说话…", flush=True)
    while True:
        arr = mic.read()
        peak_win = max(peak_win, float(np.abs(arr).max()))
        vad.accept_waveform(arr)
        if vad.is_speech_detected() and not spoke:
            spoke = True
            t_speech = time.time()
            if debug:
                print("  [LISTEN] 检测到说话…", flush=True)
        if not vad.empty():
            seg = np.asarray(vad.front.samples, dtype=np.float32)
            vad.pop()
            print(f"[LISTEN] 录到 {len(seg) / mic.sample_rate:.1f}s (peak {float(np.abs(seg).max()):.3f})")
            return seg
        now = time.time()
        if debug and now - last_report >= 2.0:
            tag = "录音中" if spoke else "等待中"
            warn = "  ←峰值<0.01，麦克风基本没拾到音！" if peak_win < 0.01 else ""
            print(f"  [LISTEN] {tag}… 近2s峰值 {peak_win:.4f}{warn}", flush=True)
            last_report, peak_win = now, 0.0
        if not spoke:
            if now - t_start > start_timeout:
                print(f"[LISTEN] {start_timeout:.0f}s 没人说话，跳过这轮")
                return np.zeros(0, dtype=np.float32)
        elif now - t_speech > max_seconds:
            print(f"[LISTEN] 说超过 {max_seconds:.0f}s 了，截断返回")
            vad.flush()
            if not vad.empty():
                seg = np.asarray(vad.front.samples, dtype=np.float32)
                vad.pop()
                return seg
            return np.zeros(0, dtype=np.float32)


# ── 麦克风（手动模式）+ 播放 ───────────────────────────────────────────────────
def _import_sd():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("需要 sounddevice：pip install sounddevice（系统还需 libportaudio2）")
    except OSError as e:
        raise RuntimeError(f"sounddevice 起不来：{e}（缺 libportaudio2？sudo apt install libportaudio2）")


def ensure_mic_default(verbose: bool = True) -> None:
    """L4T 上 USB 麦克风一拔插，PulseAudio 默认录音源就漂回板载 platform-sound（没接东西=静音）。
    这里检查一下：当前默认源若是板载/monitor，而又存在 USB 录音源，就 pactl 切过去。"""
    import shutil
    import subprocess
    if not shutil.which("pactl"):
        return
    try:
        cur = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True)
        cur_name = cur.stdout.strip()
        if not cur_name:  # 老版本 pactl 没有 get-default-source
            info = subprocess.run(["pactl", "info"], capture_output=True, text=True).stdout
            for ln in info.splitlines():
                if ln.startswith("Default Source:"):
                    cur_name = ln.split(":", 1)[1].strip()
        srcs = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True).stdout
        names = [ln.split("\t")[1] for ln in srcs.splitlines() if "\t" in ln]
        real_mics = [n for n in names if not n.endswith(".monitor") and "platform-sound" not in n]
        usb_mics = [n for n in real_mics if "usb" in n.lower()] or real_mics
        if not usb_mics:
            if verbose:
                print("[MIC] ⚠️  PulseAudio 里没看到 USB 麦克风录音源，确认麦克风插好了？", file=sys.stderr)
            return
        target = usb_mics[0]
        if cur_name == target:
            if verbose:
                print(f"[MIC] 录音源 = {target}")
            return
        subprocess.run(["pactl", "set-default-source", target], check=False)
        # 顺带把它的输出也设成默认（同一只 EarPods 收发都走它）
        out = target.replace("alsa_input.", "alsa_output.")
        if any(n == out for n in names + [s.split("\t")[1] for s in
               subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True).stdout.splitlines() if "\t" in s]):
            subprocess.run(["pactl", "set-default-sink", out], check=False)
        if verbose:
            print(f"[MIC] 默认录音源 {cur_name or '(空)'} → {target}（已切）")
    except Exception as e:  # 别因为这个挂掉
        if verbose:
            print(f"[MIC] 检查默认录音源时出错（忽略）：{e}", file=sys.stderr)


def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    sd = _import_sd()
    sd.play(np.asarray(samples, dtype=np.float32), samplerate=sample_rate)
    sd.wait()


def beep(freq: float = 880.0, dur: float = 0.12, sample_rate: int = SAMPLE_RATE,
         vol: float = 0.25) -> None:
    """短促"嘀"一声当唤醒确认音——比 VAD 的 min_speech(0.1s) 还短，响完调用方 drain() 一下 VAD 就碰不到它。"""
    try:
        sd = _import_sd()
        t = np.arange(int(dur * sample_rate)) / sample_rate
        wav = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # 两端各 5ms 淡入淡出，免得"啪"的爆音
        fade = max(1, int(0.005 * sample_rate))
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
        sd.play(wav, samplerate=sample_rate)
        sd.wait()
    except Exception:
        pass


def record_fixed(duration: float = 5.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    sd = _import_sd()
    print(f"[MIC] 录音 {duration:.0f}s ...", flush=True)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[MIC] 录音结束")
    return audio.flatten()


def record_until_enter(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    sd = _import_sd()
    input("[MIC] 按回车开始录音…")
    blocks: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}", file=sys.stderr)
        blocks.put(indata.copy())

    stop = threading.Event()
    threading.Thread(target=lambda: (input("[MIC] 录音中…按回车停止\n"), stop.set()), daemon=True).start()
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=cb):
        while not stop.is_set():
            time.sleep(0.05)
    chunks = []
    while not blocks.empty():
        chunks.append(blocks.get())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).flatten()
    print(f"[MIC] 录到 {len(audio) / sample_rate:.1f}s")
    return audio


def looks_silent(samples: np.ndarray, thresh: float = 0.01) -> bool:
    return samples.size == 0 or float(np.abs(samples).max()) < thresh


def prep_for_asr(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """喂给 ASR 前处理一下：① 峰值归一到 ~0.7（电平忽大忽小时抹平，识别更稳）
    ② 前后各补 0.1s 静音（SenseVoice 对句首句尾贴边的音不太友好）。"""
    if samples.size == 0:
        return samples
    peak = float(np.abs(samples).max())
    if 1e-4 < peak < 0.95:
        samples = samples * (0.7 / peak)
    pad = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
    return np.concatenate([pad, samples.astype(np.float32), pad])


# ── 一轮对话 ──────────────────────────────────────────────────────────────────
class Pipeline:
    """把 ASR / llm / TTS / UDP 串起来。"""

    def __init__(self, asr: ASR, *, chat: bool, tts: TTS = None,
                 udp_addr=None, session_id="voice", save_audio: bool = False):
        self.asr = asr
        self.tts = tts
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp_addr else None
        self.udp_addr = udp_addr
        self.save_audio = save_audio
        self._n = 0
        self.pm = None
        self.chat_once = None
        if chat:
            from llm import Prompt, chat_once
            self.pm = Prompt(session_id=session_id, user_name="unitree")
            self.chat_once = chat_once

    def speak(self, text: str) -> float:
        """把一句话说出来。返回"说完之前调用方还得等几秒"——本地 TTS 是阻塞的，返 0；
        走 G1 喇叭（UDP→C++ talk 节点）是异步的，返回按字数估的朗读时长，调用方该 sleep 这么久
        再去听麦克风，否则机器人自己的声音会被 VAD/KWS 当成用户输入（回声）。"""
        text = text.strip()
        if not text:
            return 0.0
        if self.udp_sock:
            self.udp_sock.sendto(text.encode("utf-8"), self.udp_addr)
        if self.tts:
            t0 = time.time()
            self.tts.say(text)
            print(f"        (本地 TTS+播放 {time.time() - t0:.2f}s)")
            return 0.0
        if self.udp_sock:
            est = 1.3 + 0.22 * len(text)   # 网络 + 云 TTS 启动 + 朗读（中文~4字/秒）的粗估
            print(f"        (→ UDP {self.udp_addr[0]}:{self.udp_addr[1]} → G1 念，约 {est:.1f}s 内不收麦)")
            return est
        return 0.0

    def transcribe(self, samples: np.ndarray) -> str:
        """识别一段语音并打印；空/太小声会说明原因。返回文本（可能是空串）。"""
        if self.save_audio and samples.size:
            path = f"/tmp/talk_heard_{self._n}.wav"
            write_wave(path, samples)
            print(f"        (录音存到 {path})")
            self._n += 1
        if looks_silent(samples):
            pk = 0.0 if samples.size == 0 else float(np.abs(samples).max())
            print(f"[识别] (空 —— peak={pk:.4f}，太小声或没拾到音)")
            return ""
        t0 = time.time()
        text = self.asr.transcribe_samples(prep_for_asr(samples))
        if text:
            print(f"[识别] {text!r}   (ASR {time.time() - t0:.2f}s, {len(samples)/SAMPLE_RATE:.1f}s 音频)")
        else:
            print(f"[识别] (ASR 没识别出内容 —— {len(samples)/SAMPLE_RATE:.1f}s 音频 peak={float(np.abs(samples).max()):.3f}；"
                  f"可能太短/口音/背景吵，加 --save-audio 听 /tmp 里的录音)")
        return text

    def handle_samples(self, samples: np.ndarray) -> float:
        return self.handle_text(self.transcribe(samples))

    def handle_text(self, text: str) -> float:
        """处理一句用户文本：跑 llm → 出声。返回 speak() 的"还得等几秒"（见上）。"""
        if not text.strip() or not self.chat_once:   # 空 / 只转写不对话
            return 0.0
        t0 = time.time()
        reply = self.chat_once(self.pm, text)
        print(f"[回复] {reply}   (LLM {time.time() - t0:.2f}s)")
        return self.speak(reply)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="语音助手：唤醒 → ASR → llm → TTS")
    p.add_argument("wav", nargs="?", help="要转写的 wav 文件")
    p.add_argument("--mic", action="store_true", help="手动录一段（回车开始/再回车停止）")
    p.add_argument("--seconds", type=float, default=None, help="手动模式改成固定录 N 秒")
    p.add_argument("--chat", action="store_true", help="转写结果丢给 llm.py 跑对话")
    p.add_argument("--loop", action="store_true", help="连续对话（隐含 --mic --chat），每轮回车录一句")
    p.add_argument("--wake", action="store_true", help="免手模式：等唤醒词 → VAD 录音 → ASR → llm → TTS")
    p.add_argument("--hear", action="store_true", help="调试：一直 VAD 断句 → ASR 打印你说了啥（不唤醒、不对话、不出声）")
    p.add_argument("--wake-words", default=",".join(DEFAULT_WAKE_WORDS), help="逗号分隔的唤醒词")
    p.add_argument("--wake-threshold", type=float, default=0.25, help="唤醒触发阈值（越大越难触发）")
    p.add_argument("--listen-seconds", type=float, default=12.0, help="唤醒后单句最长录音秒数")
    p.add_argument("--no-tts", action="store_true", help="不加载本地 TTS（仍会 UDP 发文本；配合 --g1 / C++ talk 节点）")
    p.add_argument("--tts", action="store_true", help="非 loop/wake 模式也朗读回复")
    p.add_argument("--g1", action="store_true",
                   help="用 G1 自带喇叭朗读：经 ~/unitree_sdk2/build/bin/talk 节点（先在另一终端跑 `./talk eth0`）；隐含 --no-tts")
    p.add_argument("--debug", action="store_true", help="多打状态：麦克风峰值、VAD 起停等")
    p.add_argument("--save-audio", action="store_true", help="把每段录到的语音存到 /tmp/talk_heard_N.wav")
    p.add_argument("--udp-host", default="127.0.0.1", help="把回复文本 UDP 发给 C++ talk 的主机")
    p.add_argument("--udp-port", type=int, default=9872, help="同上，端口（0 = 不发）")
    args = p.parse_args()

    if args.loop or args.wake:
        args.mic = True
        args.chat = True
    if args.g1:
        args.no_tts = True
    if not args.wav and not args.mic and not args.hear:
        p.print_help()
        sys.exit(1)

    want_tts = (args.tts or args.loop or args.wake) and not args.no_tts
    if args.g1:
        print(f"[G1] 回复将经 UDP {args.udp_host}:{args.udp_port} 交给 C++ talk 节点 → G1 喇叭朗读")
        print(f"     （记得在另一终端跑：~/unitree_sdk2/build/bin/talk <网口，如 eth0>）")

    # 用麦克风的模式：先确保 PulseAudio 默认录音源指向 USB 麦克风，并打印实际用的设备
    if args.mic or args.hear:
        ensure_mic_default()
        try:
            sd = _import_sd()
            dev = sd.query_devices(sd.default.device[0])
            print(f"[MIC] 输入设备: {dev['name']}  (默认采样率 {dev['default_samplerate']:.0f}Hz)")
        except Exception as e:
            print(f"[MIC] 列设备失败（忽略）：{e}", file=sys.stderr)

    asr = ASR()
    tts = TTS() if want_tts else None
    udp_addr = (args.udp_host, args.udp_port) if (args.chat and args.udp_port) else None
    pipe = Pipeline(asr, chat=args.chat, tts=tts, udp_addr=udp_addr, save_audio=args.save_audio)

    # ① 文件模式
    if args.wav:
        if not Path(args.wav).exists():
            sys.exit(f"文件不存在: {args.wav}")
        text = asr.transcribe_file(args.wav)
        print(f"[识别] {text!r}")
        pipe.handle_text(text)
        return

    # ⓪ 纯听写调试模式：一条常开麦克风流 + 一个 VAD，连续断句 → ASR 打印
    if args.hear:
        mic = MicStream()
        vad = make_vad()
        print("=== 听写调试：随便说话，它会断句并打印识别结果；Ctrl-C 退出 ===")
        last_report = time.time()
        peak_win = 0.0
        try:
            while True:
                arr = mic.read()
                peak_win = max(peak_win, float(np.abs(arr).max()))
                vad.accept_waveform(arr)
                if not vad.empty():
                    seg = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    print(f"[LISTEN] 录到 {len(seg) / mic.sample_rate:.1f}s (peak {float(np.abs(seg).max()):.3f})")
                    pipe.transcribe(seg)
                if time.time() - last_report >= 3.0:
                    warn = "  ←峰值<0.01，麦克风基本没拾到音！" if peak_win < 0.01 else ""
                    print(f"  [LISTEN] 监听中… 近3s峰值 {peak_win:.4f}{warn}", flush=True)
                    last_report, peak_win = time.time(), 0.0
        except KeyboardInterrupt:
            print("\n再见")
        finally:
            mic.close()
        return

    # ② 免手唤醒模式
    if args.wake:
        wake = WakeWord([w.strip() for w in args.wake_words.split(",") if w.strip()],
                        threshold=args.wake_threshold)
        vad = make_vad()
        mic = MicStream()

        def settle(busy: float) -> None:
            """机器人说完之前别收麦：等估算的朗读时长 + 丢掉这期间麦克风缓冲（含机器人自己的声音/回声）。"""
            if busy and busy > 0:
                time.sleep(busy)
            mic.drain()

        settle(pipe.speak("我准备好了"))
        print("=== 免手模式：说唤醒词开始；Ctrl-C 退出 ===")
        try:
            while True:
                print(f"\n[KWS] 等待唤醒词（{args.wake_words}）…", flush=True)
                hit = wake.wait(mic, debug=args.debug)
                print(f"[KWS] 唤醒：{hit}")
                beep()                   # 唤醒确认音（本地短"嘀"，不走 TTS，没回声问题）
                mic.drain()              # 把刚那声"嘀"从缓冲里清掉
                samples = vad_capture(vad, mic, max_seconds=args.listen_seconds, debug=args.debug)
                settle(pipe.handle_samples(samples))   # 回复念完前别收麦（G1 喇叭时按估时长 sleep）
        except KeyboardInterrupt:
            print("\n再见")
        finally:
            mic.close()
        return

    # ③ 手动循环 / 单次
    def one_round():
        samples = record_fixed(args.seconds) if args.seconds else record_until_enter()
        pipe.handle_samples(samples)

    if args.loop:
        print("=== 连续对话：每轮回车开始说话、说完再回车；Ctrl-C 退出 ===")
        try:
            while True:
                one_round()
        except (KeyboardInterrupt, EOFError):
            print("\n再见")
    else:
        one_round()


if __name__ == "__main__":
    main()
