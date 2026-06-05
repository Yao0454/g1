"""
gesture.py — 纯手势识别入口（无人物跟随、无语音）。

只起 Vision 模块，关掉跟随只保留手势检测。检到挥手/握手/双手举起
就经 bridge.send_arm() 把动作 id 丢给 C++ g1_node。

C++ 端先在另一终端跑：
    ~/unitree_sdk2/build/bin/g1_node eth0

用法：
    python gesture.py                       # 默认本机 g1_node
    python gesture.py --bridge-host 192.168.x.x
    python gesture.py --cooldown 6          # 改两次动作的最小间隔
    python gesture.py --stream-port 0       # 关掉 MJPEG 调试流
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core import Bridge, MjpegStream, Vision


def main() -> None:
    p = argparse.ArgumentParser(description="只跑手势识别（无跟随/无语音）")
    p.add_argument("--bridge-host", default="127.0.0.1", help="C++ g1_node 地址")
    p.add_argument("--bridge-port", type=int, default=9870, help="C++ g1_node 端口")
    p.add_argument("--stream-port", type=int, default=6769, help="MJPEG 调试流端口（0 关）")
    p.add_argument("--cooldown", type=float, default=4.0, help="两次手势动作的最小间隔(s)")
    p.add_argument("--max-dist", type=float, default=3.0, help="只锁这个距离以内的人（m）")
    p.add_argument("--log-frames", action="store_true",
                   help="把每帧推理结果存到 data/snapshots/（事后排查识别问题）")
    p.add_argument("--log-frames-every", type=float, default=2.0, help="快照间隔(s)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    bridge = Bridge(host=args.bridge_host, port=args.bridge_port, verbose=args.verbose)
    print(f"[GESTURE] bridge → {args.bridge_host}:{args.bridge_port} (只发 ARM)")

    stream = MjpegStream(port=args.stream_port) if args.stream_port else None
    snapshot_dir = (
        Path(__file__).resolve().parent / "data" / "snapshots" if args.log_frames else None
    )
    vision = Vision(
        bridge,
        stream=stream,
        follow=False,
        gesture=True,
        action_cooldown=args.cooldown,
        max_dist=args.max_dist,
        snapshot_dir=snapshot_dir,
        snapshot_every=args.log_frames_every,
        verbose=args.verbose,
    )
    try:
        vision.run()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
