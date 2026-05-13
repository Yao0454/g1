"""
llm.py — 命令行 ollama REPL。

用 core.Chat（持久化会话历史到 ./data/session/<id>.json）。
    python llm.py                # 默认 session=test, user=unitree
    python llm.py --session foo  # 换会话
    python llm.py --model qwen3:8b
"""
from __future__ import annotations

import argparse

from core import Chat


def main() -> None:
    p = argparse.ArgumentParser(description="ollama 对话 REPL")
    p.add_argument("--session", default="test")
    p.add_argument("--user", default="unitree")
    p.add_argument("--model", default="qwen3:8b")
    args = p.parse_args()

    chat = Chat(session_id=args.session, user_name=args.user, model=args.model)
    print(f"=== ollama REPL  session={args.session}  model={args.model}  (输入 q 退出) ===")
    while True:
        try:
            text = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "q":
            return
        if not text:
            continue
        try:
            reply = chat.reply(text)
        except Exception as e:
            print(f"调用失败: {e}")
            continue
        print(f"AI: {reply}")


if __name__ == "__main__":
    main()
