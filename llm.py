from __future__ import annotations  # 兼容 Python 3.8：让 list[dict] 之类的注解延迟求值

import json
import os
from datetime import datetime
from pathlib import Path

# ollama 跑在本机 11434，别让系统代理把 localhost 拦走
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import ollama

MAX_ITERATIONS: int = 10
AGENT_NAME: str = "宇树 G1 机器人"
ERROR_MESSAGES = {
    "max_iter": f"已达到最大迭代次数（{MAX_ITERATIONS} 次），任务终止",
    "load_failed": "历史记录加载失败：{reason}",
    "save_failed": "历史记录保存失败：{reason}",
}


class Prompt:
    def __init__(self, session_id: str = "default", user_name: str = ""):
        self.session_id = session_id
        self.user_name = user_name
        self._SYSTEM_TEMPLATE: str = "你是一个非常友善的宇树 G1 机器人，你的概括能力很强，你每次回答都是一句话，不超过20个字"

        # 会话历史存储目录
        self.session_path = Path.cwd() / "data" / "session"
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.history = self.load_history()

    def __del__(self):
        self.save_history(self.history)

    def _history_path(self) -> Path:
        """返回当前 session 的历史文件路径"""
        return self.session_path / f"{self.session_id}.json"

    def _format_extra_section(self, extra_context: str = "") -> str:
        """构建 System Prompt 的可选补充段落"""
        parts = []
        if self.user_name:
            parts.append(f"\n## 当前用户\n你正在服务的用户是：{self.user_name}")
        if extra_context:
            parts.append(f"\n## 额外上下文\n{extra_context}")
        return "".join(parts)

    # ── 历史记录 ──────────────────────────────────────────

    def load_history(self) -> list[dict]:
        """
        从文件加载历史消息列表。

        Returns:
            消息列表，格式为 [{"role": "user"/"assistant", "content": "..."}, ...]
            文件不存在或解析失败时返回空列表。
        """
        path = self._history_path()
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            print(ERROR_MESSAGES["load_failed"].format(reason=e))
            return []

    def save_history(self, history: list[dict]) -> None:
        """
        将消息列表持久化到文件。

        Args:
            history: 消息列表，格式同 load_history 返回值。
        """
        try:
            with open(self._history_path(), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(ERROR_MESSAGES["save_failed"].format(reason=e))

    def update_history(self, user_input: str, response: str) -> None:
        """
        追加一轮对话到历史记录（拿到回复后再一起写，避免中途失败留下半条）。

        Args:
            user_input: 用户输入。
            response:   助手回复。
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        self.save_history(self.history)

    def clear_history(self) -> None:
        """清空当前 session 的历史记录"""
        self.save_history([])

    def build_system_prompt(self, extra_context: str = "") -> str:
        """
        构建完整的 System Prompt。

        Args:
            extra_context: 额外上下文信息，追加到 System Prompt 末尾。

        Returns:
            格式化后的 System Prompt 字符串。
        """
        return self._SYSTEM_TEMPLATE + self._format_extra_section(extra_context)

    def build_messages(self, user_input: str) -> list[dict]:
        """拼出本轮发给模型的完整消息：system + 历史 + 本轮 user。

        注意：system 每次现拼，不存进 history；本轮 user 也先不写 history，
        等 update_history 成功拿到回复后再一起写。
        """
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})
        return messages


def chat_once(pm: "Prompt", text_in: str, model: str = "qwen3:8b") -> str:
    """跑一轮对话：输入文本 → 模型回复（同时写进历史）。供 talk.py 复用。"""
    messages = pm.build_messages(text_in)
    response = ollama.chat(model=model, messages=messages, think=False)
    reply = response["message"]["content"]
    pm.update_history(text_in, reply)
    return reply


def main() -> None:
    pm = Prompt(session_id="test", user_name="unitree")

    while True:
        text_in = input("User: ").strip()
        if text_in == "q":
            break
        if not text_in:
            continue
        try:
            reply = chat_once(pm, text_in)
        except Exception as e:
            print(f"调用失败: {e}")
            continue
        print(f"AI: {reply}")


if __name__ == "__main__":
    main()
