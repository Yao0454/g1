"""
core/bridge.py — 跟 C++ g1_node 通信的单端口 UDP 客户端。

线协议（一个 UDP 端口，默认 9870）：
    byte 0 = type
      0x01 MOVE → +3 floats (dist, err_norm, blocked)        → 13 字节
      0x02 ARM  → +1 uint8  (action_id)                      →  2 字节
      0x03 TTS  → +UTF-8 text                                → 1+N 字节

C++ 那边 ~/unitree_sdk2/source/g1_node/main.cpp 按首字节 switch。改字段记得两边同步。
"""
from __future__ import annotations

import socket
import struct

MSG_MOVE: int = 0x01
MSG_ARM:  int = 0x02
MSG_TTS:  int = 0x03


class Bridge:
    """一个 UDP socket 发所有命令到 g1_node。线程安全（sendto 在 Linux 上是原子的）。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9870, verbose: bool = False):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.verbose = verbose

    # ── 跟随：底盘速度的输入（视觉算出来的人的距离/水平偏差/是否被挡）─────────
    def send_move(self, dist: float, err_norm: float, blocked: bool) -> None:
        pkt = struct.pack("<Bfff", MSG_MOVE, float(dist), float(err_norm),
                          1.0 if blocked else 0.0)
        self.sock.sendto(pkt, self.addr)
        if self.verbose:
            print(f"[BRIDGE] MOVE dist={dist:.2f} err={err_norm:+.2f} blk={blocked}")

    # ── 手臂动作：action_id（15=举手 26=挥手 27=握手 99=收回，看 G1 SDK 文档）─
    def send_arm(self, action_id: int) -> None:
        pkt = struct.pack("<BB", MSG_ARM, int(action_id) & 0xFF)
        self.sock.sendto(pkt, self.addr)
        if self.verbose:
            print(f"[BRIDGE] ARM  action={action_id}")

    # ── TTS：让 G1 自带喇叭念这段中文 ───────────────────────────────────────
    def send_tts(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        pkt = bytes([MSG_TTS]) + text.encode("utf-8")
        self.sock.sendto(pkt, self.addr)
        if self.verbose:
            print(f"[BRIDGE] TTS  {text!r}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass
