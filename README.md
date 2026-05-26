# 扫地机器人知识问答助手

基于 LlamaIndex RAG 框架的智能问答系统，帮助用户快速解决扫地机器人使用中的各种问题。

## 技术栈

- **框架**: LlamaIndex 0.14 (RAG)
- **LLM**: Ollama + qwen2.5:7b (本地运行)
- **Embedding**: 阿里云百炼 DashScope text-embedding-v2
- **向量库**: ChromaDB
- **UI**: Gradio

## 快速开始

### 1. 配置 API Key

编辑 `.env` 文件，填入阿里云百炼的 API Key：

```
DASHSCOPE_API_KEY=your-key-here
```

> 从 https://bailian.console.aliyun.com/ 获取

### 2. 准备知识库

将您的知识文件（支持 `.txt`、`.pdf`、`.docx` 等格式）放入 `knowledge_base/` 目录。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 构建知识库索引

```bash
python build_index.py
```

### 5. 启动

**Web UI模式**（推荐）：

```bash
python app.py
```

打开浏览器访问 http://localhost:7860

**CLI模式**：

```bash
python main.py
```

## 项目结构

```
├── .env                 # API Key 配置
├── .gitignore
├── requirements.txt     # 依赖
├── build_index.py       # ChromaDB 索引构建
├── query_engine.py      # RAG 查询引擎
├── app.py               # Gradio Web UI
├── main.py              # CLI 入口
├── knowledge_base/      # 用户上传的知识库文件
└── index/               # ChromaDB 持久化目录（自动生成）
```

## 支持的文件格式

`knowledge_base/` 支持放入多种文件格式，包括：
- `.txt` — 纯文本
- `.pdf` — PDF 文档
- `.docx` — Word 文档
- `.md` — Markdown

放入文件后重新运行 `python build_index.py` 即可更新索引。
