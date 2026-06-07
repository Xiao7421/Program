"""扫地机器人知识问答助手 - Gradio Web UI (含文件上传)"""
import os
import shutil
import logging
import gradio as gr
from query_engine import CleaningRobotQA
from build_index import build_index as rebuild_kb_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KB_DIR = "knowledge_base"
INDEX_DIR = "./index"
qa = CleaningRobotQA()


# ── 工具函数 ──────────────────────────────────────────────

def list_kb_files():
    if not os.path.exists(KB_DIR):
        return []
    return sorted(os.listdir(KB_DIR))


def files_to_markdown(file_list):
    if not file_list:
        return "（暂无文件，请上传）"
    return "\n".join(f"- {f}" for f in file_list)


# ── UI 回调 ───────────────────────────────────────────────

def upload_files(files):
    if not files:
        return files_to_markdown(list_kb_files())

    os.makedirs(KB_DIR, exist_ok=True)
    saved = []
    for f in files:
        dst = os.path.join(KB_DIR, os.path.basename(f.name))
        shutil.copy2(f.name, dst)
        saved.append(os.path.basename(f.name))

    return files_to_markdown(list_kb_files())


def rebuild_index(progress=gr.Progress()):
    if not list_kb_files():
        return "⚠ knowledge_base 为空，请先上传文件"

    progress(0, desc="清理旧索引...")
    if os.path.exists(INDEX_DIR):
        qa.close()  # 释放 ChromaDB 文件锁
        shutil.rmtree(INDEX_DIR)

    progress(0.3, desc="构建向量索引...")
    try:
        rebuild_kb_index()
    except Exception as e:
        logger.exception("索引重建失败")
        return f"❌ 构建失败: {e}"

    progress(0.8, desc="加载新索引...")
    qa.reload()

    progress(1.0, desc="完成")
    doc_count = len(list_kb_files())
    return f"✅ 重建完成！已索引 {doc_count} 个文件"


def chat_fn(message, history):
    if not qa.is_ready:
        return "⚠ 索引未构建。请先上传知识文件并点击「重建索引」。"
    if not message.strip():
        return "请输入您的问题。"
    try:
        response, sources = qa.query(message)
        return f"{response}\n\n{sources}" if sources else response
    except Exception as e:
        logger.exception("查询失败")
        return f"查询出错: {str(e)}"


# ── UI 构建 ───────────────────────────────────────────────

css = """
footer { display: none !important; }
"""

with gr.Blocks(title="扫地机器人问答助手") as demo:
    gr.Markdown("# 🤖 扫地机器人知识问答助手")

    with gr.Row(equal_height=False):
        # ── 左侧：聊天 ──
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_fn,
                title="",
                description="问任何关于扫地机器人的问题 — 使用、故障、维护...",
                examples=[
                    "扫地机器人无法开机怎么办？",
                    "滚刷缠绕头发了怎么清理？",
                    "多久换一次滤网？",
                    "为什么总是找不到充电座？",
                ],
            )

        # ── 右侧：文件管理 ──
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📁 知识库管理")

            upload = gr.File(
                label="上传知识文件",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".docx", ".md"],
            )

            file_list = gr.Markdown(files_to_markdown(list_kb_files()))

            rebuild_btn = gr.Button("🔄 重建索引", variant="primary")

            status = gr.Markdown(
                "✅ 就绪" if qa.is_ready else "⚠ 未构建索引（请上传文件并重建）"
            )

    # ── 事件 ──
    upload.upload(
        fn=upload_files,
        inputs=upload,
        outputs=file_list,
    )

    rebuild_btn.click(
        fn=rebuild_index,
        inputs=[],
        outputs=status,
    )


if __name__ == "__main__":
    kb_count = len(list_kb_files())
    print("=" * 50)
    print(f"  Web UI: http://localhost:7860")
    print(f"  知识库: {kb_count} 个文件")
    print(f"  索引:   {'就绪' if qa.is_ready else '未构建'}")
    print("=" * 50)
    demo.launch(server_name="0.0.0.0", server_port=7860, theme="soft", css=css)
