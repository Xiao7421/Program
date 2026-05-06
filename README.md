# SuperBizAgent

> 企业级智能对话与 AIOps 运维助手，集成 RAG 知识库问答和自动化故障诊断能力

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 核心特性

- **智能对话** - 基于 LangChain 的多轮对话，支持流式输出
- **RAG 问答** - 向量检索增强，支持文档上传、自动索引、知识库动态更新
- **AIOps 诊断** - Plan-Execute-Replan 自动故障诊断与根因分析
- **Web 界面** - 现代化前端，支持快速问答和流式对话模式
- **MCP 集成** - 通过 Model Context Protocol 接入日志查询和监控工具

## 技术栈

- **后端框架**: FastAPI
- **AI 框架**: LangChain + LangGraph
- **LLM**: 阿里云 DashScope (通义千问)
- **向量数据库**: Milvus
- **工具协议**: MCP (Model Context Protocol)

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- 阿里云 DashScope API Key

### 安装和启动

#### Linux/macOS

```bash
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 安装依赖
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. 配置环境变量
# 编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env

# 4. 一键初始化（Docker + 服务 + 文档）
make init

# 5. 启动服务
make start
```

#### Windows

```powershell
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 创建虚拟环境并安装依赖
pip install uv
uv venv
.venv\Scripts\activate
uv pip install -e .

# 3. 配置环境变量
notepad .env

# 4. 启动 Milvus 向量数据库
docker compose -f vector-database.yml up -d

# 5. 等待 Milvus 启动（约 10 秒）
timeout /t 10

# 6. 启动 MCP 服务（新窗口）
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py

# 7. 启动主服务（新窗口）
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 8. 上传知识库文档（新窗口，等待服务启动后执行）
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键脚本**

```powershell
.\start-windows.bat   # 启动所有服务
.\stop-windows.bat    # 停止所有服务
```

### 访问服务

- Web 界面: http://localhost:9900
- API 文档: http://localhost:9900/docs

## API 接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并索引文档 |
| 健康检查 | GET | `/api/health` | 服务状态检查 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer
```

## 项目结构

```
super_biz_agent_py/
├── app/                                    # 应用核心
│   ├── main.py                             # FastAPI 入口
│   ├── config.py                           # 配置管理
│   ├── api/                                # API 路由
│   ├── services/                           # 业务服务层
│   ├── agent/                              # Agent 模块（MCP 客户端、AIOps）
│   ├── models/                             # 数据模型
│   ├── tools/                              # Agent 工具集
│   ├── core/                               # 核心组件（LLM 工厂、Milvus 客户端）
│   └── utils/                              # 工具类
├── static/                                 # Web 前端
├── mcp_servers/                            # MCP 服务器
├── aiops-docs/                             # 运维知识库
├── .env                                    # 环境变量配置
├── Makefile                                # 项目管理命令
├── vector-database.yml                     # Milvus Docker Compose 配置
└── pyproject.toml                          # 项目配置与依赖
```

## 配置说明

编辑 `.env` 文件：

```bash
# DashScope 配置（必填）
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# RAG 配置
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100
```

## AIOps 智能运维

基于 Plan-Execute-Replan 模式实现自动故障诊断：

```
Planner 制定诊断计划
    ↓
Executor 执行步骤（调用 MCP 工具）
    ↓
Replanner 评估结果，决定继续 / 调整 / 生成报告
    ↓
输出诊断报告（根因分析 + 运维建议）
```

### 快速测试

```bash
# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test"}' \
  --no-buffer
```

## 开发指南

```bash
make init              # 一键初始化
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务
make install-dev       # 安装开发依赖
make format            # 格式化代码
make lint              # 代码检查
```

## 常见问题

### API Key 错误

检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确配置。

### Milvus 连接失败

```bash
# 确保 Docker 已启动
docker ps | grep milvus

# 重启 Milvus
docker compose -f vector-database.yml restart
```

### 端口被占用

```bash
# Linux/macOS
lsof -i :9900

# Windows
netstat -ano | findstr :9900
taskkill /F /PID <PID>
```

## 许可证

MIT License
