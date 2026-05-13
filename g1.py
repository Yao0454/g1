"""
g1.py — 三件事一起跑：跟随 + 手势 + 语音对话。

架构：
    主线程：Vision（30 FPS 相机+YOLO-pose）→ bridge.send_move / send_arm
    后台线程：Voice（唤醒→VAD→ASR→Chat→TTS）→ bridge.send_tts
    bridge 是同一个 UDP socket，发到本机 g1_node (默认 127.0.0.1:9870)
    g1_node 那边按首字节分派给 LocoClient / ArmActionClient / AudioClient

C++ 端：先在另一终端跑
    ~/unitree_sdk2/build/bin/g1_node eth0

用法：
    python g1.py                            # 全开
    python g1.py --no-voice                 # 只视觉
    python g1.py --no-vision                # 只语音
    python g1.py --no-gesture --no-voice    # 只跟随
    python g1.py --bridge-host 192.168.x.x  # 跨机
"""
from __future__ import annotations

import argparse
import sys
import time

from core import Bridge, Chat, MjpegStream, Vision, Voice


def main() -> None:
    p = argparse.ArgumentParser(description="G1 跟随 + 手势 + 语音 一起跑")
    p.add_argument("--bridge-host", default="127.0.0.1", help="C++ g1_node 地址")
    p.add_argument("--bridge-port", type=int, default=9870, help="C++ g1_node 端口")
    p.add_argument("--no-vision", action="store_true", help="不开视觉")
    p.add_argument("--no-voice", action="store_true", help="不开语音")
    p.add_argument("--no-follow", action="store_true", help="开视觉但不跟随（只手势）")
    p.add_argument("--no-gesture", action="store_true", help="开视觉但不识手势（只跟随）")
    p.add_argument("--no-chat", action="store_true", help="开语音但不调 LLM（只听写打印）")
    p.add_argument("--local-tts", action="store_true",
                   help="语音回复走本地 EarPods 而非 G1 喇叭（调试用）")
    p.add_argument("--wake-words", default=None, help="逗号分隔的唤醒词")
    p.add_argument("--wake-threshold", type=float, default=0.25)
    p.add_argument("--listen-seconds", type=float, default=12.0)
    p.add_argument("--stream-port", type=int, default=6769, help="MJPEG 调试流端口（0 关）")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.no_vision and args.no_voice:
        sys.exit("--no-vision 和 --no-voice 不能同时给（什么都不开）")

    bridge = Bridge(host=args.bridge_host, port=args.bridge_port, verbose=args.verbose)
    print(f"[G1] bridge → {args.bridge_host}:{args.bridge_port} (一个 UDP 端口走三种消息)")

    voice = None
    if not args.no_voice:
        chat = None if args.no_chat else Chat(session_id="g1", user_name="unitree")
        wake = [w.strip() for w in args.wake_words.split(",")] if args.wake_words else None
        voice = Voice(bridge, chat=chat,
                      wake_words=wake,
                      wake_threshold=args.wake_threshold,
                      listen_seconds=args.listen_seconds,
                      local_tts=args.local_tts,
                      verbose=args.verbose)
        voice.start()  # 后台线程

    # 视觉跑主线程（相机/YOLO 比较重，留主线程）
    if not args.no_vision:
        stream = MjpegStream(port=args.stream_port) if args.stream_port else None
        vision = Vision(bridge, stream=stream,
                        follow=not args.no_follow,
                        gesture=not args.no_gesture,
                        verbose=args.verbose)
        try:
            vision.run()   # 阻塞，Ctrl-C 跳出
        finally:
            if voice is not None:
                voice.stop()
            bridge.close()
    else:
        # 只语音模式：主线程啥也不干，等 Ctrl-C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[G1] Ctrl-C 退出")
        finally:
            if voice is not None:
                voice.stop()
            bridge.close()


if __name__ == "__main__":
    main()
