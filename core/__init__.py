"""g1 core — 跟随 / 手势 / 语音 的可复用类。

惯例：
    from core import Bridge, Vision, Voice, Chat, MjpegStream

或者单独按需 import：
    from core.bridge import Bridge
    ...
"""

from .bridge import MSG_ARM, MSG_MOVE, MSG_TTS, Bridge
from .chat import Chat
from .stream import MjpegStream
from .vision import Vision
from .voice import Voice

__all__ = [
    "Bridge",
    "Vision",
    "Voice",
    "Chat",
    "MjpegStream",
    "MSG_MOVE",
    "MSG_ARM",
    "MSG_TTS",
]
