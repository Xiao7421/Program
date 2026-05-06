# 扫地机器人问答助手

基于 LangChain Agent + RAG 的扫地机器人智能问答系统，支持流式输出和前端交互。

## 功能特性

- **RAG 检索增强生成**：基于 ChromaDB 向量数据库，对扫地机器人相关知识文档进行语义检索，结合大模型生成回答
- **ReAct Agent**：使用 LangChain Agent 框架，支持工具调用（天气查询、用户信息获取、外部数据读取等）
- **流式输出**：后端通过 SSE（Server-Sent Events）实现流式响应
- **动态提示词切换**：通过中间件机制，根据上下文动态切换系统提示词（普通问答 / 报告生成）
- **前端界面**：基于 Vue 3 + Vite 构建的 Web 前端

## 项目结构

```
├── app_server.py          # FastAPI 后端入口
├── agent/
│   └── tools/
│       ├── react_agent.py     # ReAct Agent 核心实现
│       ├── agent_tools.py     # Agent 工具定义（RAG检索、天气、用户数据等）
│       └── middleware.py      # Agent 中间件（工具监控、提示词动态切换）
├── rag/
│   ├── rag_service.py         # RAG 检索+总结服务
│   └── vector_store.py        # 向量库管理（文档加载、分片、去重）
├── model/
│   └── factory.py             # 模型工厂（ChatTongyi / DashScope Embeddings）
├── config/                    # YAML 配置文件（rag、chroma、agent、prompts）
├── prompts/                   # 提示词模板（主提示词、RAG提示词、报告提示词）
├── data/                      # 知识库文档（PDF/TXT）
├── utils/                     # 工具类（配置加载、文件处理、日志、路径）
├── chroma_db/                 # ChromaDB 持久化存储
├── logs/                      # 日志输出目录
└── app_ui/                    # Vue 3 前端项目
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| Agent 框架 | LangChain (ReAct Agent) |
| 向量数据库 | ChromaDB |
| 大模型 | 通义千问 (ChatTongyi) |
| Embedding | DashScope Embeddings |
| 前端 | Vue 3 + Vite + marked |

## 环境准备

1. Python 3.10+
2. Node.js 18+（前端）
3. 阿里云 DashScope API Key

## 配置

设置环境变量：

```bash
export DASHSCOPE_API_KEY=your_api_key
```

各模块配置通过 `config/` 目录下的 YAML 文件管理：

- `config/rag.yml` — RAG 模型名称、API 配置
- `config/chroma.yml` — 向量库集合名、分片参数、数据路径
- `config/agent.yml` — Agent 外部数据路径等
- `config/prompts.yml` — 提示词模板路径

## 启动

### 后端

```bash
pip install -r requirements.txt
python app_server.py
```

后端默认运行在 `http://localhost:8000`，API 文档：`http://localhost:8000/docs`

### 前端

```bash
cd app_ui
npm install
npm run dev
```

### 加载知识库

```bash
python -m rag.vector_store
```

将 `data/` 目录下的文档加载到 ChromaDB 向量库中，自动根据文件 MD5 去重。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 流式问答接口，SSE 响应 |
| GET | `/api/health` | 健康检查 |
