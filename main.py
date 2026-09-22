"""Minimal terminal chat loop. Run: python main.py"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv

load_dotenv()

from bot import ask  # noqa: E402  (import after load_dotenv on purpose)


def main():
    thread_id = str(uuid.uuid4())
    print("Weather-Advisory Support Bot (terminal mode). Type 'exit' to quit.")
    print(f"[session id: {thread_id}]\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        try:
            response = ask(thread_id, user_message)
            print(f"\nBot: {response.answer}")
            if response.primary_sop_id:
                print(f"     [Policy: {response.primary_sop_id}]")
        except RuntimeError as e:
            if "high demand" in str(e):
                print(f"\nBot: {e}")
            else:
                raise
        print()


if __name__ == "__main__":
    main()
