"""扫地机器人知识问答助手 - CLI 交互模式"""
import sys
import logging
from query_engine import CleaningRobotQA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        qa = CleaningRobotQA()
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    print("=" * 50)
    print("  扫地机器人知识问答助手 (CLI 模式)")
    print("  输入问题开始问答，输入 /clear 清除历史，输入 exit 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n🙋 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if user_input == "/clear":
            qa.clear_memory()
            print("对话历史已清除。")
            continue

        print("\n🤖 助手: ", end="", flush=True)
        response, sources = qa.query(user_input)
        print(response)
        if sources:
            print(f"\n{sources}")


if __name__ == "__main__":
    main()
