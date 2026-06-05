# 企业级RAG应用设计文档 — 基于LlamaIndex深度构建

> 日期: 2026-06-05
> 状态: 已审核
> 技术栈: LlamaIndex 0.11+ / FastAPI / Milvus / DeepSeek / DashScope

---

## 1. 项目概述

基于LlamaIndex框架构深度构建企业级RAG（Retrieval-Augmented Generation）应用，充分运用LlamaIndex 6大核心能力：Workflows事件驱动、Ingestion Pipeline智能摄取、Router Query Engine智能路由、Callback Manager可观测性、Composable Index组合索引、LlamaHub数据连接器生态。

**核心功能**：
- 上传 txt/pdf/markdown 文件到知识库
- 基于知识库的多轮对话问答
- 4种高级检索策略可选切换
- 流式响应输出
- 简易前端可视化界面

---

## 2. 技术选型

| 维度 | 选型 | 说明 |
|---|---|---|
| RAG框架 | LlamaIndex >= 0.11.0 | 核心框架，提供全部RAG能力 |
| LLM | DeepSeek (deepseek-chat) | 通过 `llama-index-llms-deepseek` 接入 |
| 文本嵌入 | DashScope text-embedding-v2 | 通过 `llama-index-embeddings-dashscope` 接入 |
| 向量数据库 | Milvus Standalone (Docker) | 通过 `llama-index-vector-stores-milvus` 接入，端口19530 |
| 会话记忆 | ChatMemoryBuffer | LlamaIndex内置，token_limit=3000 |
| 后端 | FastAPI + Uvicorn | 轻量级，支持SSE流式响应 |
| 前端 | 纯HTML + CSS + JS | 无构建步骤，FastAPI静态文件服务 |
| 文件解析 | LlamaIndex Readers | SimpleDirectoryReader / 自定义reader |

---

## 3. 架构设计

### 3.1 整体架构：模块化服务分层

```
┌─────────────────────────────────────────────────────┐
│                   前端 (HTML+CSS+JS)                  │
│         聊天界面 | 文件上传 | 知识库管理 | 策略选择      │
├─────────────────────────────────────────────────────┤
│                FastAPI 路由层 + SSE                    │
│    /api/upload | /api/documents | /api/chat | /api/config │
├─────────────────────────────────────────────────────┤
│                  Service 服务层                       │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  🔄 RAGWorkflow (LlamaIndex Workflows)        │   │
│  │  ┌────────┐  ┌────────┐  ┌───────┐  ┌─────┐ │   │
│  │  │Rewrite │→│Retrieve │→│Rerank │→│Gen  │ │   │
│  │  └────────┘  └────────┘  └───────┘  └─────┘ │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌────────────────┐  ┌──────────────────────────┐   │
│  │ Ingestion      │  │ Router Query Engine       │   │
│  │ Pipeline       │  │ (智能路由: 向量/摘要/HyDE) │   │
│  │ (增量摄取+缓存)│  │                           │   │
│  └────────────────┘  └──────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  📊 Callback Manager (可观测性)               │   │
│  │  Token消耗 | 检索延迟 | LLM调用追踪           │   │
│  └──────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  MilvusVectorStore | DashScopeEmbedding | DeepSeek  │
└─────────────────────────────────────────────────────┘
```

### 3.2 项目目录结构

```
rag-enterprise/
├── backend/
│   ├── main.py                      # FastAPI 入口 + 路由 + 静态文件服务
│   ├── config.py                    # 配置管理 (pydantic-settings, .env)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py           # DeepSeek LLM 单例初始化
│   │   ├── embedding_service.py     # DashScope Embedding 单例初始化
│   │   ├── document_service.py      # 文件上传/解析/Ingestion Pipeline
│   │   ├── index_service.py         # Milvus 向量索引管理
│   │   └── query_service.py         # RAGWorkflow + Router + 4种策略
│   ├── models/
│   │   └── schemas.py               # Pydantic 请求/响应数据模型
│   └── uploads/                     # 临时文件存储目录
├── frontend/
│   ├── index.html                   # 主页面（聊天+知识库管理）
│   ├── css/
│   │   └── style.css                # 样式文件
│   └── js/
│       └── app.js                   # 前端交互逻辑
├── .env.example                     # 环境变量模板
├── requirements.txt                 # Python 依赖清单
└── docker-compose.yml               # Milvus 容器编排
```

---

## 4. LlamaIndex 核心能力运用

### 4.1 Ingestion Pipeline（智能摄取管线）

替代手动的 "读取→分块→嵌入→入库" 流程，利用LlamaIndex内置的IngestionPipeline：

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

pipeline = IngestionPipeline(
    transformations=[
        SentenceWindowNodeParser(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        ),
        DashScopeEmbedding(model_name="text-embedding-v2"),
    ],
    vector_store=milvus_vector_store,
    cache=IngestionCache(),  # 增量摄取：已处理的文档不重复处理
)
```

**核心优势**：
- 内置缓存机制，支持增量更新
- 自动去重，避免重复处理已入库文档
- transformations可组合，灵活切换分块策略

### 4.2 Workflows 事件驱动系统

用LlamaIndex 0.11+的Workflow系统替代硬编码pipeline，实现可插拔的检索流水线：

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event

class QueryRewriteEvent(Event):
    query: str

class RetrieveEvent(Event):
    query: str

class RerankEvent(Event):
    nodes: list

class RAGWorkflow(Workflow):
    @step
    async def rewrite(self, ev: StartEvent) -> QueryRewriteEvent:
        """Step 1: 查询重写 — 将口语化提问转为检索友好形式"""
        rewritten = await rewrite_query(ev.query, self.llm)
        return QueryRewriteEvent(query=rewritten)

    @step
    async def retrieve(self, ev: QueryRewriteEvent) -> RetrieveEvent:
        """Step 2: 检索 — 支持HyDE/Sentence Window/基础模式"""
        nodes = await self._retrieve_with_strategy(ev.query, self.strategy)
        return RetrieveEvent(query=ev.query, nodes=nodes)

    @step
    async def rerank(self, ev: RetrieveEvent) -> RerankEvent:
        """Step 3: Re-ranking — CrossEncoder精排"""
        reranked = self.reranker.postprocess_nodes(ev.nodes, ev.query)
        return RerankEvent(nodes=reranked[:self.top_n])

    @step
    async def generate(self, ev: RerankEvent) -> StopEvent:
        """Step 4: 生成 — 结合会话记忆 + System Prompt"""
        response = await self._generate_with_memory(ev.query, ev.nodes)
        return StopEvent(result=response)
```

**核心优势**：
- 每个步骤独立可测试，松耦合
- 可动态增删步骤（如关闭reranking只需跳过该step）
- 事件驱动，天然支持异步并行

### 4.3 Router Query Engine（智能路由）

根据问题类型自动选择最佳检索路径：

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleObjectSelector

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="精确检索",
    description="适用于查找具体事实、数据、定义类问题"
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="摘要概览",
    description="适用于需要全局了解、总结归纳类问题"
)

router_engine = RouterQueryEngine(
    selector=LLMSingleObjectSelector(llm=deepseek_llm),
    query_engine_tools=[vector_tool, summary_tool],
    verbose=True
)
```

**核心优势**：
- LLM自动判断走哪条检索路径，无需硬编码
- 可轻松扩展新检索路径（如Knowledge Graph、SQL查询）

### 4.4 Callback Manager（可观测性）

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager
```

**自动追踪**：
- LLM调用次数和token消耗
- 每次检索的延迟和命中数量
- Embedding调用次数
- 完整的调用链路（可用于前端展示"思考过程"）

### 4.5 Composable Index（组合索引）

```python
# 不同类型文档使用不同索引策略
# PDF/Markdown → VectorStoreIndex (精确检索)
# 长文档 → SummaryIndex (全局摘要)

from llama_index.core import VectorStoreIndex, SummaryIndex

vector_index = VectorStoreIndex.from_vector_store(milvus_store)
summary_index = SummaryIndex(nodes=document_summaries)
```

### 4.6 LlamaHub 数据连接器

```python
# 预置支持的文件类型
from llama_index.core import SimpleDirectoryReader

# 未来可扩展 LlamaHub 连接器:
# from llama_index.readers.web import SimpleWebPageReader
# from llama_index.readers.database import DatabaseReader
# from llama_index.readers.notion import NotionPageReader
```

---

## 5. 后端设计

### 5.1 API 端点

| 方法 | 端点 | 功能 | 请求体 | 响应 |
|---|---|---|---|---|
| `POST` | `/api/upload` | 上传文件 | `multipart/form-data` (file) | `{file_id, filename, chunks_count, status}` |
| `GET` | `/api/documents` | 获取文档列表 | - | `[{file_id, filename, upload_time, chunks_count}]` |
| `DELETE` | `/api/documents/{file_id}` | 删除文档及向量数据 | - | `{status: "ok"}` |
| `POST` | `/api/chat` | 发送提问 | `{question: str, strategy: str}` | SSE 流式响应 |
| `GET` | `/api/config` | 获取可用配置 | - | `{strategies: [...], default: "rerank"}` |
| `GET` | `/api/stats` | 获取运行统计 | - | `{documents, chunks, queries, tokens_used}` |

### 5.2 服务层设计

#### `llm_service.py`
- DeepSeek LLM 单例管理
- 支持 temperature 等参数配置
- 通过 `llama-index-llms-deepseek` 包接入

#### `embedding_service.py`
- DashScope text-embedding-v2 单例管理
- 通过 `llama-index-embeddings-dashscope` 包接入
- 统一 Settings.embed_model 配置

#### `document_service.py`
- 文件上传验证（类型、大小限制）
- 支持 .txt / .pdf / .markdown(.md) 三种格式
- 使用 LlamaIndex SimpleDirectoryReader + 自定义reader解析
- 集成 Ingestion Pipeline 完成自动分块+嵌入+入库
- metadata 注入：source_file, file_type, upload_time, chunk_index

#### `index_service.py`
- Milvus Standalone 连接管理 (host:port)
- Collection 创建/删除/查询
- 基于 `llama-index-vector-stores-milvus` 的 VectorStoreIndex
- 文档删除时同步清理对应向量数据

#### `query_service.py`（核心）
- RAGWorkflow 事件驱动检索流水线
- 4种检索策略实现：
  - **BASIC**: 直接向量相似度检索 (top_k=20)
  - **HYDE**: HyDEQueryTransform 生成假设文档后检索
  - **SENTENCE_WINDOW**: SentenceWindowNodeParser + MetadataReplacementPostProcessor
  - **RE_RANKING**: SentenceTransformerRerank / CrossEncoderRerank 精排
- 查询重写（始终启用）
- ChatMemoryBuffer 多轮对话管理 (token_limit=3000)
- System Prompt 工程（角色设定 + 引用格式 + 约束规则）
- SSE 流式响应生成

### 5.3 数据模型 (`schemas.py`)

```python
class ChatRequest(BaseModel):
    question: str
    strategy: str = "rerank"  # basic | hyde | window | rerank

class ChatEvent(BaseModel):
    type: str       # thinking | sources | token | done | error
    content: str

class DocumentInfo(BaseModel):
    file_id: str
    filename: str
    file_type: str
    chunks_count: int
    upload_time: datetime

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    chunks_count: int
    status: str

class AppConfig(BaseModel):
    strategies: list[str]
    default_strategy: str
    llm_model: str
    embedding_model: str
```

### 5.4 检索流水线详细步骤

```
用户提问
   │
   ▼
Step 1: 查询重写 (Query Rewriting)
   │  用DeepSeek将口语化提问重写为检索友好形式
   │  例: "这个产品咋用?" → "产品的使用方法和操作指南"
   ▼
Step 2: 策略检索 (按所选策略执行)
   │  ├─ BASIC: 直接向量相似度 top_k=20
   │  ├─ HYDE: LLM生成假设答案 → embedding假设文档 → 检索
   │  ├─ SENTENCE_WINDOW: 检索小chunk → 扩展上下文窗口
   │  └─ RE_RANKING: 初检top_k=20 → CrossEncoder精排 → top_n=5
   ▼
Step 3: 上下文组装
   │  ChatMemoryBuffer 历史消息拼接
   │  System Prompt 注入
   ▼
Step 4: 流式生成
   │  DeepSeek streaming response
   │  SSE事件推送: thinking → sources → tokens → done
   ▼
返回前端
```

### 5.5 System Prompt 模板

```
你是一个专业的企业知识库助手。请基于提供的上下文文档回答用户问题。

规则：
1. 仅基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"文档中未找到相关信息"
3. 回答中引用来源时，使用 [来源: 文件名] 格式标注
4. 使用Markdown格式组织回答，合理使用标题、列表、代码块

上下文文档：
{context}

会话历史：
{chat_history}

用户问题：{question}
```

---

## 6. 前端设计

### 6.1 页面布局

```
┌──────────────────────────────────────────────────────┐
│  🏢 企业级RAG知识库助手                    [⚙️ 设置]   │
├────────────────┬─────────────────────────────────────┤
│                │                                     │
│  📁 知识库管理  │      💬 对话区域                      │
│                │                                     │
│  [📤 上传文件]  │  ┌─────────────────────────────┐    │
│                │  │ 🤖 你好！我是知识库助手       │    │
│  文件列表:      │  └─────────────────────────────┘    │
│  📄 产品手册.pdf │  ┌─────────────────────────────┐    │
│  📄 API文档.md  │  │ 👤 请介绍产品的核心功能      │    │
│  📄 会议记录.txt │  └─────────────────────────────┘    │
│                │  ┌─────────────────────────────┐    │
│  [🗑️ 删除选中]  │  │ 🤖 根据文档，核心功能包括... │    │
│                │  │    [来源: 产品手册.pdf]       │    │
│  ──────────── │  └─────────────────────────────┘    │
│  检索策略:     │                                     │
│  ○ 基础检索    │                                     │
│  ● Re-ranking  │                                     │
│  ○ HyDE       │                                     │
│  ○ 句子窗口    │                                     │
│                │                                     │
├────────────────┴─────────────────────────────────────┤
│  [📎]  输入问题...                          [发送 ▶]  │
└──────────────────────────────────────────────────────┘
```

### 6.2 核心交互

1. **文件上传**
   - 左侧面板拖拽或点击上传区域
   - 支持 .txt / .pdf / .md 文件
   - 上传时显示进度条和解析状态
   - 上传成功后自动刷新文件列表和chunk数量

2. **对话交互**
   - SSE (Server-Sent Events) 流式接收回答
   - 实时逐token显示AI回答
   - Markdown渲染（引入 marked.js CDN）
   - 代码高亮（引入 highlight.js CDN）
   - 回答底部展示引用来源列表（文件名+相关度得分）

3. **策略切换**
   - 左侧单选按钮组，4种策略可选
   - 切换后下次提问生效
   - 每种策略有简短说明tooltip

4. **知识库管理**
   - 文件列表展示（文件名、类型图标、上传时间、chunk数）
   - 支持勾选删除
   - 删除时同步清理向量数据

### 6.3 技术方案

- **CSS**：Flexbox/Grid布局，CSS变量实现主题切换
- **JS**：原生 `fetch` API + `EventSource` 处理SSE
- **外部CDN**：marked.js（Markdown渲染）、highlight.js（代码高亮）
- **无构建步骤**：浏览器直接访问 FastAPI 托管的静态文件

---

## 7. 配置与部署

### 7.1 环境变量 (`.env`)

```env
# DeepSeek LLM
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_MODEL=deepseek-chat

# 百炼 DashScope Embedding
DASHSCOPE_API_KEY=sk-your-dashscope-key
DASHSCOPE_EMBED_MODEL=text-embedding-v2

# Milvus Standalone
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_docs

# RAG 参数
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=20
TOP_N=5
CHAT_TOKEN_LIMIT=3000
DEFAULT_STRATEGY=rerank

# 文件上传限制
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./uploads
```

### 7.2 依赖清单 (`requirements.txt`)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
python-dotenv>=1.0.0
pydantic-settings>=2.0.0

# LlamaIndex Core + Plugins
llama-index>=0.11.0
llama-index-llms-deepseek
llama-index-embeddings-dashscope
llama-index-vector-stores-milvus
llama-index-postprocessor-cross-encoder-rerank
pymilvus>=2.4.0
```

### 7.3 Docker Compose (`docker-compose.yml`)

```yaml
services:
  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
    command: etcd --advertise-client-urls=http://127.0.0.1:2379 --listen-client-urls=http://0.0.0.0:2379 --data-dir=/etcd

  milvus-minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data

  milvus-standalone:
    image: milvusdb/milvus:v2.4-latest
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - milvus-etcd
      - milvus-minio
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
```

### 7.4 启动流程

```bash
# 1. 启动 Milvus 容器集群
docker compose up -d

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY

# 4. 启动后端服务
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. 访问应用
# 浏览器打开 http://localhost:8000
```

---

## 8. 文件解析策略

| 文件类型 | Reader | 分块策略 |
|---|---|---|
| `.txt` | 直接读取 (Python内置) | SentenceSplitter / SentenceWindowNodeParser |
| `.pdf` | LlamaIndex PDFReader (pypdf) | SentenceSplitter / SentenceWindowNodeParser |
| `.markdown` / `.md` | LlamaIndex MarkdownReader | SentenceSplitter / SentenceWindowNodeParser，保留标题层级metadata |

所有文件解析后统一注入以下metadata：
- `source_file`: 原始文件名
- `file_type`: 文件类型 (txt/pdf/md)
- `upload_time`: 上传时间戳
- `file_id`: 文件唯一标识
- `chunk_index`: 分块序号

---

## 9. 错误处理

| 场景 | 处理方式 |
|---|---|
| 文件上传失败 | 返回400 + 错误信息，不入库 |
| 不支持的文件类型 | 返回415，前端提示支持的格式 |
| 文件过大 | 返回413，默认限制50MB |
| Milvus连接失败 | 启动时检测，失败则返回503 |
| DeepSeek API超时 | 重试3次，失败后SSE推送error事件 |
| DashScope API失败 | 重试2次，文件上传返回错误 |
| 知识库为空时提问 | 正常回答，提示"当前知识库为空，请先上传文档" |
| 会话历史超限 | ChatMemoryBuffer自动截断旧消息 |

---

## 10. 评估集成（预留）

基于PDF3中的Ragas评估框架，预留评估接口（MVP不实现，但架构支持）：

```python
# 预留评估端点
# GET /api/eval/datasets     - 测试集管理
# POST /api/eval/run         - 运行评估
# GET /api/eval/results      - 评估结果

# 预留指标:
# - faithfulness (忠实度)
# - answer_relevancy (回答相关性)
# - context_precision (上下文精确度)
# - context_recall (上下文召回率)
```
