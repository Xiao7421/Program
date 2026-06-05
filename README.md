# 🏢 企业级RAG知识库助手

基于 **LlamaIndex** 深度构建的企业级 RAG（检索增强生成）应用，支持多种高级检索策略、流式对话、文件上传与知识库管理。

![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![Milvus](https://img.shields.io/badge/Milvus-2.4-orange)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple)

## ✨ 核心功能

- 📤 **文件上传** — 支持 `.txt` / `.pdf` / `.md`，拖拽或点击上传，自动解析入库
- 🔍 **3种检索策略** — 基础向量检索、HyDE 假设文档检索、句子窗口检索
- 📊 **Re-ranking 重排序** — CrossEncoder 精排，可叠加在任意检索策略上
- 🔄 **查询重写** — 自动将口语化提问转为检索友好形式
- 💬 **多轮对话** — ChatMemoryBuffer 上下文记忆，连续追问
- ⚡ **流式响应** — SSE 逐 token 输出 + Markdown 渲染 + 代码高亮
- 📎 **引用来源** — 回答附带来源文件和相关度得分
- 🌙 **暗色主题** — 美观的前端界面，响应式布局

## 🏗️ 架构

```
┌──────────────────────────────────────────────────┐
│              前端 (HTML + CSS + JS)               │
│       聊天界面 | 文件上传 | 知识库管理 | 策略选择    │
├──────────────────────────────────────────────────┤
│             FastAPI 路由层 + SSE 流式              │
├──────────────────────────────────────────────────┤
│               Service 服务层                      │
│  ┌────────────────────────────────────────────┐  │
│  │  DocumentService — IngestionPipeline       │  │
│  │  QueryService — 重写→检索→Rerank→生成      │  │
│  │  IndexService — Milvus 向量索引管理         │  │
│  │  LLM/Embedding — DeepSeek / DashScope      │  │
│  └────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────┤
│  MilvusVectorStore | DashScopeEmbedding | DeepSeek │
└──────────────────────────────────────────────────┘
```

### LlamaIndex 核心能力运用

| 能力 | 用途 | 代码位置 |
|---|---|---|
| **IngestionPipeline** | 自动分块 + 嵌入 + 入库，内置缓存 | `document_service.py` |
| **SentenceWindowNodeParser** | 句子窗口分块，保留上下文窗口 metadata | `document_service.py` |
| **ChatMemoryBuffer** | 多轮对话记忆，自动截断旧消息 | `query_service.py` |
| **HyDE** | 生成假设答案文档后检索，提升语义匹配 | `query_service.py` |
| **MetadataReplacementPostProcessor** | 句子窗口检索时扩展上下文 | `query_service.py` |
| **SentenceTransformerRerank** | CrossEncoder 交叉编码器精排 | `query_service.py` |
| **PromptTemplate** | System Prompt 工程，角色 + 规则 + 引用格式 | `query_service.py` |
| **Callback Manager** | LlamaDebugHandler 可观测性追踪 | `index_service.py` |

## 📁 项目结构

```
d:/llamaindex/
├── backend/
│   ├── main.py                      # FastAPI 入口 + 路由 + SSE 流式
│   ├── config.py                    # Pydantic Settings (.env 加载)
│   ├── services/
│   │   ├── llm_service.py           # DeepSeek LLM 单例
│   │   ├── embedding_service.py     # DashScope Embedding 单例
│   │   ├── index_service.py         # Milvus 向量索引管理
│   │   ├── document_service.py      # 文件上传/解析/IngestionPipeline
│   │   └── query_service.py         # 核心 RAG: 重写→3策略→Rerank→记忆→生成
│   ├── models/
│   │   └── schemas.py               # Pydantic 请求/响应数据模型
│   ├── uploads/                     # 上传文件暂存
│   └── data/                        # 文档注册表 (JSON)
├── frontend/
│   ├── index.html                   # 主页面
│   ├── css/
│   │   └── style.css                # 暗色主题样式
│   └── js/
│       └── app.js                   # SSE 流式聊天 + 文件上传 + 策略切换
├── .env.example                     # 环境变量模板
├── requirements.txt                 # Python 依赖
└── docker-compose.yml               # Milvus 容器编排 (参考)
```

## 🚀 快速开始

### 前置条件

- Python 3.10+
- Milvus 已运行（Docker 或本地）
- DeepSeek API Key
- 百炼 DashScope API Key

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
DEEPSEEK_API_KEY=sk-你的deepseek密钥
DASHSCOPE_API_KEY=sk-你的百炼密钥
```

### 3. 启动 Milvus（如果还没运行）

```bash
docker compose up -d
```

### 4. 启动后端

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 访问应用

浏览器打开 **http://localhost:8000**

## 📡 API 接口

| 方法 | 端点 | 功能 |
|---|---|---|
| `POST` | `/api/upload` | 上传文件 (multipart/form-data) |
| `GET` | `/api/documents` | 获取文档列表 |
| `DELETE` | `/api/documents/{file_id}` | 删除文档及向量数据 |
| `POST` | `/api/chat` | 发送提问 (SSE 流式响应) |
| `GET` | `/api/config` | 获取可用配置 |
| `GET` | `/api/stats` | 获取运行统计 |

### 请求示例

**上传文件：**
```bash
curl -X POST http://localhost:8000/api/upload -F "file=@your_document.pdf"
```

**发送提问：**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "请介绍产品的核心功能", "strategy": "basic", "use_rerank": true}'
```

## 🔍 检索策略说明

### 基础检索 (basic)
直接向量相似度检索，速度最快，适合事实类问题。

### HyDE 假设文档检索 (hyde)
先让 LLM 生成一个"假设性答案"，再用答案的 embedding 去检索，提升语义匹配度。适合模糊或开放性问题。

### 句子窗口检索 (window)
检索小块文本后，自动扩展为包含前后 3 句的上下文窗口。兼顾精确匹配和完整上下文。

### Re-ranking 重排序 (可叠加)
使用 CrossEncoder (`ms-marco-MiniLM-L-6-v2`) 对初检结果精排，可叠加在任意检索策略上。

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 512 | 文本分块大小（SentenceWindow 模式下为窗口大小） |
| `CHUNK_OVERLAP` | 50 | 分块重叠大小 |
| `TOP_K` | 20 | 初检返回数量 |
| `TOP_N` | 5 | Rerank 后保留数量 |
| `CHAT_TOKEN_LIMIT` | 3000 | 会话记忆 token 上限 |
| `MAX_FILE_SIZE_MB` | 50 | 单文件上传大小限制 |
| `EMBEDDING_DIM` | 1536 | 嵌入向量维度 (DashScope v2) |
| `DEFAULT_STRATEGY` | basic | 默认检索策略 |
| `DEFAULT_USE_RERANK` | true | 默认是否启用 Rerank |

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| RAG 框架 | LlamaIndex >= 0.11.0 |
| LLM | DeepSeek (deepseek-chat) via OpenAI-compatible API |
| 文本嵌入 | DashScope text-embedding-v2 (百炼) |
| 向量数据库 | Milvus Standalone 2.4 (Docker) |
| 重排序 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 后端 | FastAPI + Uvicorn |
| 前端 | 纯 HTML + CSS + JS (marked.js + highlight.js) |

## 📄 License

MIT
