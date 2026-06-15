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

from core.skills import list_skills_detail
from core.tools import tools

MAX_ROUNDS: int = 5


class Chat:
    SYSTEM_PROMPT = "你是一个非常友善的宇树 G1 机器人，你的概括能力很强，你每次回答都是一句话，不超过20个字"

    def __init__(
        self,
        session_id: str = "g1",
        user_name: str = "unitree",
        model: str = "qwen3:8b",
        data_dir: Path | None = None,
    ):
        self.session_id = session_id
        self.user_name = user_name
        self.model = model
        self.data_dir = data_dir or Path.cwd() / "data" / "session"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = self._load()

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
                json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[CHAT] 历史保存失败：{e}")

    def clear(self) -> None:
        self.history = []
        self._save()

    def _system(self, extra: str = "") -> str:
        s = self.SYSTEM_PROMPT
        if self.user_name:
            s += f"\n\n## 当前用户\n你正在服务的用户是：{self.user_name}"
        if extra:
            s += f"\n\n## 额外上下文\n{extra}"
        skills: list[tuple[str, str]] = list_skills_detail()
        if skills:
            s += "\n\n## 可用技能（需要时调用 load_skill 加载详细说明）\n"
            s += "\n".join(f"- {name}: {desc}" for name, desc in skills)
        return s

    def reply(self, user_text: str, extra_context: str = "") -> str:
        """跑一轮对话：输入文本 → 模型回复（同时追加历史）。"""
        messages = [{"role": "system", "content": self._system(extra_context)}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        reply_text = ""  # 先初始化为空
        for _ in range(MAX_ROUNDS):  # 循环多轮，直到 no tool_call 为止
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools.schemas,
                think=False,
            )  # 喂给 llm
            msg = resp["message"]
            messages.append(dict(msg))
            tool_calls = msg.get("tool_calls")  # 获取 tool_calls
            if not tool_calls:
                reply_text = msg["content"]
                break
            for tc in tool_calls:  # 执行 tool_calls
                fn = tc["function"]
                result = tools.call(fn["name"], fn["arguments"])
                messages.append(
                    {"role": "tool", "name": fn["name"], "content": str(result)}
                )
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        self._save()
        return reply_text
