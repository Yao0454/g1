"""
talk.py — 纯语音工具（语音不带视觉/动作）。

默认（--wake）：等唤醒词 → 录音 → ASR → ollama → 让 G1 喇叭念。
回复经 bridge 发到 C++ g1_node（默认 127.0.0.1:9870），需要先在另一终端跑：
    ~/unitree_sdk2/build/bin/g1_node eth0

用法：
    python talk.py                         # 默认 wake 模式（chat + G1 喇叭）
    python talk.py --hear                  # 调试：纯听写打印，不唤醒/不对话/不出声
    python talk.py --no-chat               # 只识别打印，不调 LLM
    python talk.py --local-tts             # 回复走本地 EarPods 而非 G1 喇叭
    python talk.py --wake-words "你好机器人,小宇" --wake-threshold 0.2
"""
from __future__ import annotations

import argparse
import time

from core import Bridge, Chat, Voice


def main() -> None:
    p = argparse.ArgumentParser(description="纯语音工具（基于 core.Voice）")
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", type=int, default=9870)
    p.add_argument("--hear", action="store_true", help="纯听写：VAD 断句 + ASR 打印，不唤醒/不对话/不出声")
    p.add_argument("--no-chat", action="store_true", help="唤醒后只识别打印，不调 LLM")
    p.add_argument("--local-tts", action="store_true", help="回复经本地 EarPods 而非 G1 喇叭")
    p.add_argument("--wake-words", default=None, help="逗号分隔的唤醒词")
    p.add_argument("--wake-threshold", type=float, default=0.25)
    p.add_argument("--listen-seconds", type=float, default=12.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    bridge = Bridge(host=args.bridge_host, port=args.bridge_port, verbose=args.debug)
    chat = None if args.no_chat else Chat(session_id="talk", user_name="unitree")
    wake = [w.strip() for w in args.wake_words.split(",")] if args.wake_words else None

    voice = Voice(bridge, chat=chat,
                  wake_words=wake,
                  wake_threshold=args.wake_threshold,
                  listen_seconds=args.listen_seconds,
                  local_tts=args.local_tts,
                  verbose=args.debug)

    if args.hear:
        try:
            voice.hear()
        finally:
            bridge.close()
        return

    voice.start()  # 后台线程
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n再见")
    finally:
        voice.stop()
        bridge.close()


if __name__ == "__main__":
    main()
