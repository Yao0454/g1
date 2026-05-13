"""
core/chat.py — ollama 对话封装。

用法：
    chat = Chat(session_id="g1", user_name="unitree")
    reply = chat.reply("你好")
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ollama 跑在本机 11434，别让系统代理把 localhost 拦走
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import ollama  # noqa: E402


class Chat:
    SYSTEM_PROMPT = "你是一个非常友善的宇树 G1 机器人，你的概括能力很强，你每次回答都是一句话，不超过20个字"

    def __init__(self, session_id: str = "g1", user_name: str = "unitree",
                 model: str = "qwen3:8b", data_dir: Path | None = None):
        self.session_id = session_id
        self.user_name = user_name
        self.model = model
        self.data_dir = (data_dir or Path.cwd() / "data" / "session")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = self._load()

    # ── 历史持久化 ───────────────────────────────────────────────────────────
    def _path(self) -> Path:
        return self.data_dir / f"{self.session_id}.json"

    def _load(self) -> list[dict]:
        p = self._path()
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[CHAT] 历史加载失败：{e}")
            return []

    def _save(self) -> None:
        try:
            self._path().write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            print(f"[CHAT] 历史保存失败：{e}")

    def clear(self) -> None:
        self.history = []
        self._save()

    # ── 调用 ────────────────────────────────────────────────────────────────
    def _system(self, extra: str = "") -> str:
        s = self.SYSTEM_PROMPT
        if self.user_name:
            s += f"\n\n## 当前用户\n你正在服务的用户是：{self.user_name}"
        if extra:
            s += f"\n\n## 额外上下文\n{extra}"
        return s

    def reply(self, user_text: str, extra_context: str = "") -> str:
        """跑一轮对话：输入文本 → 模型回复（同时追加历史）。"""
        messages = [{"role": "system", "content": self._system(extra_context)}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        resp = ollama.chat(model=self.model, messages=messages, think=False)
        reply = resp["message"]["content"]
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self._save()
        return reply
