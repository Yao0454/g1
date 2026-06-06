"""
core/api.py — Flutter 前端 WebSocket 控制接口（端口 8765）。

消息协议（JSON）：
  Flutter → Jetson（命令）：
    {"type":"move","dist":2.5,"err":-0.1}       遥控移动（follow 关时生效）
    {"type":"arm","action":"wave"}               手臂动作 wave/handshake/raise/retract
    {"type":"tts","text":"你好"}                  让机器人说话
    {"type":"toggle","feature":"follow","enabled":true}
    {"type":"toggle","feature":"gesture","enabled":false}
    {"type":"toggle","feature":"voice","enabled":true}
    {"type":"stop"}                              紧急停止

  Jetson → Flutter（推送）：
    {"type":"state","follow":true,"gesture":true,"voice":true}
    {"type":"log","level":"info","msg":"...","ts":"12:34:56"}
    {"type":"asr","text":"你好机器人"}
    {"type":"llm","text":"你好！"}

遥控移动说明：
    g1_node C++ 按"目标距离"逻辑控速；follow 关闭时前端接管 MOVE 包：
      向前：dist = 1.0 + y * 1.5  (y ∈ [0,1] → dist ∈ [1.0, 2.5])
      向后：dist = 1.0 + y * 1.0  (y ∈ [-1,0] → dist ∈ [0.0, 1.0])
      转向：err_norm = joystick_x  (-1=左转, +1=右转)
    由 Flutter 端 Joystick 组件计算后直接发。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable, Optional, Set

log = logging.getLogger(__name__)

ARM_NAME_TO_ID: dict[str, int] = {
    "raise": 15,
    "wave": 26,
    "handshake": 27,
    "retract": 99,
}


class ApiServer:
    """WebSocket API 服务端，守护线程 + 独立 asyncio 事件循环。"""

    def __init__(self, bridge, port: int = 8765):
        self._bridge = bridge
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set = set()

        # 当前模式状态
        self.follow: bool = True
        self.gesture: bool = True
        self.voice: bool = True

        # g1.py 注册：api 收到 toggle 时同步通知 vision/voice 对象
        self.on_toggle: Optional[Callable[[str, bool], None]] = None

    # ─── 线程安全推送（任意线程调） ──────────────────────────────────────────

    def push_log(self, level: str, msg: str) -> None:
        self._emit({"type": "log", "level": level, "msg": msg,
                    "ts": time.strftime("%H:%M:%S")})

    def push_asr(self, text: str) -> None:
        self._emit({"type": "asr", "text": text})

    def push_llm(self, text: str) -> None:
        self._emit({"type": "llm", "text": text})

    def push_state(self) -> None:
        self._emit({"type": "state",
                    "follow": self.follow,
                    "gesture": self.gesture,
                    "voice": self.voice})

    def _emit(self, msg: dict) -> None:
        if not self._loop:
            return
        data = json.dumps(msg, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    # ─── 内部 async ──────────────────────────────────────────────────────────

    async def _broadcast(self, data: str) -> None:
        dead: Set = set()
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _handle(self, ws) -> None:
        self._clients.add(ws)
        log.info("[API] 前端连接 %s", getattr(ws, "remote_address", "?"))
        try:
            await ws.send(json.dumps({
                "type": "state",
                "follow": self.follow,
                "gesture": self.gesture,
                "voice": self.voice,
            }))
            async for raw in ws:
                try:
                    await self._dispatch(json.loads(raw))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            log.info("[API] 前端断开")

    async def _dispatch(self, m: dict) -> None:
        t = m.get("type")
        if t == "move":
            self._bridge.send_move(
                float(m.get("dist", 1.0)),
                float(m.get("err", 0.0)),
                float(m.get("blocked", 0.0)),
            )
        elif t == "arm":
            action = m.get("action", "retract")
            aid = ARM_NAME_TO_ID.get(action, 99) if isinstance(action, str) else int(action)
            self._bridge.send_arm(aid)
        elif t == "tts":
            text = str(m.get("text", "")).strip()
            if text:
                self._bridge.send_tts(text)
        elif t == "toggle":
            feat = m.get("feature", "")
            enabled = bool(m.get("enabled", True))
            if feat in ("follow", "gesture", "voice"):
                setattr(self, feat, enabled)
                if self.on_toggle:
                    self.on_toggle(feat, enabled)
                await self._broadcast(json.dumps({
                    "type": "state",
                    "follow": self.follow,
                    "gesture": self.gesture,
                    "voice": self.voice,
                }))
        elif t == "stop":
            # dist=TARGET_DIST=1.0 → vx=0; err=0 → vyaw=0 → SetVelocity(0,0,0)
            self._bridge.send_move(1.0, 0.0, 0.0)

    # ─── 启动 ────────────────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        """守护线程里启动 WS 服务器，立即返回线程对象。"""
        def _run() -> None:
            import websockets  # 懒加载，没装时不影响其他模块
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _serve() -> None:
                async with websockets.serve(self._handle, "0.0.0.0", self._port):
                    print(f"[API] WebSocket 监听 0.0.0.0:{self._port}  Flutter 连这里")
                    await asyncio.Future()

            self._loop.run_until_complete(_serve())

        t = threading.Thread(target=_run, daemon=True, name="api-ws")
        t.start()
        return t
