# Enterprise RAG Application — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an enterprise RAG application with LlamaIndex featuring file upload (txt/pdf/md), multi-strategy retrieval, reranking, multi-turn chat with streaming, and a simple web UI.

**Architecture:** Modular service-layer architecture — FastAPI backend serves both REST API and static frontend. Service layer separates concerns: LLM/embedding singletons, Milvus vector store management, document ingestion pipeline (LlamaIndex IngestionPipeline), and query pipeline (rewrite → retrieve → rerank → generate) with SSE streaming. All LlamaIndex settings configured globally via `Settings`.

**Tech Stack:** LlamaIndex >= 0.11.0, FastAPI, Milvus Standalone (Docker), DeepSeek LLM (OpenAI-compatible), DashScope text-embedding-v2, SentenceTransformerRerank, ChatMemoryBuffer, pure HTML/CSS/JS frontend.

**Spec:** `docs/superpowers/specs/2026-06-05-enterprise-rag-llamaindex-design.md`

---

## File Structure

```
rag-enterprise/                       # project root (d:/llamaindex)
├── backend/
│   ├── main.py                       # FastAPI app, routes, static file mount
│   ├── config.py                     # Pydantic Settings, env var loading
│   ├── services/
│   │   ├── __init__.py               # empty
│   │   ├── llm_service.py            # DeepSeek LLM singleton
│   │   ├── embedding_service.py      # DashScope embedding singleton
│   │   ├── index_service.py          # Milvus connection, collection mgmt
│   │   ├── document_service.py       # Upload validation, parse, ingest
│   │   └── query_service.py          # Rewrite, 3 strategies, rerank, prompt, stream
│   ├── models/
│   │   ├── __init__.py               # empty
│   │   └── schemas.py               # Pydantic request/response models
│   ├── data/                         # document registry JSON (auto-created)
│   └── uploads/                      # uploaded files (auto-created)
├── frontend/
│   ├── index.html                    # Main page: sidebar + chat area
│   ├── css/
│   │   └── style.css                 # Layout, theme, responsive
│   └── js/
│       └── app.js                    # Chat SSE, upload, strategy switch
├── .env.example                      # Env var template with placeholder keys
├── requirements.txt                  # All Python dependencies pinned
└── docker-compose.yml                # Milvus standalone (etcd + minio + milvus)
```

---

## Task 1: Project Scaffold & Infrastructure

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `backend/uploads/.gitkeep`
- Create: `backend/data/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```txt
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
python-dotenv>=1.0.0
pydantic-settings>=2.0.0

# LlamaIndex Core + Plugins
llama-index>=0.11.0
llama-index-llms-openai-like
llama-index-embeddings-dashscope
llama-index-vector-stores-milvus
pymilvus>=2.4.0

# Reranker
sentence-transformers>=2.6.0

# File parsing
pypdf>=4.0.0
```

- [ ] **Step 2: Create .env.example**

```env
# DeepSeek LLM
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# DashScope Embedding (百炼)
DASHSCOPE_API_KEY=sk-your-dashscope-key

# Milvus Standalone
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_docs

# RAG Parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=20
TOP_N=5
CHAT_TOKEN_LIMIT=3000

# File Upload
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./uploads

# Defaults
DEFAULT_STRATEGY=basic
DEFAULT_USE_RERANK=true
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.18
    container_name: milvus-etcd
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - etcd_data:/etcd

  milvus-minio:
    image: minio/minio:latest
    container_name: milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data --console-address ":9001"
    volumes:
      - minio_data:/minio_data

  milvus-standalone:
    image: milvusdb/milvus:v2.4-latest
    container_name: milvus-standalone
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - milvus-etcd
      - milvus-minio
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    volumes:
      - milvus_data:/var/lib/milvus

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p backend/services backend/models backend/uploads backend/data
mkdir -p frontend/css frontend/js
touch backend/uploads/.gitkeep backend/data/.gitkeep
touch backend/services/__init__.py backend/models/__init__.py
```

- [ ] **Step 5: Add .gitignore**

Create `.gitignore`:

```
.env
__pycache__/
*.pyc
backend/uploads/*
!backend/uploads/.gitkeep
backend/data/*
!backend/data/.gitkeep
.venv/
*.egg-info/
```

- [ ] **Step 6: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 7: Start Milvus**

Run: `docker compose up -d`
Expected: Three containers start — etcd, minio, milvus-standalone. Port 19530 is accessible.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: project scaffold with docker-compose, requirements, env template"
```

---

## Task 2: Configuration & Data Models

**Files:**
- Create: `backend/config.py`
- Create: `backend/models/schemas.py`

- [ ] **Step 1: Create backend/config.py**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # DeepSeek LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # DashScope Embedding
    dashscope_api_key: str = ""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_docs"

    # RAG Parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 20
    top_n: int = 5
    chat_token_limit: int = 3000

    # File Upload
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # Defaults
    default_strategy: str = "basic"
    default_use_rerank: bool = True

    model_config = {
        "env_file": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".env"
        ),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2: Create backend/models/schemas.py**

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    strategy: str = "basic"       # basic | hyde | window
    use_rerank: bool = True


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


class DeleteResponse(BaseModel):
    status: str
    file_id: str


class AppConfig(BaseModel):
    strategies: list[str]
    default_strategy: str
    rerank_enabled: bool
    llm_model: str
    embedding_model: str


class AppStats(BaseModel):
    documents: int
    chunks: int
    milvus_connected: bool
```

- [ ] **Step 3: Commit**

```bash
git add backend/config.py backend/models/
git commit -m "feat: add configuration and Pydantic data models"
```

---

## Task 3: LLM & Embedding Services

**Files:**
- Create: `backend/services/llm_service.py`
- Create: `backend/services/embedding_service.py`

- [ ] **Step 1: Create backend/services/llm_service.py**

DeepSeek is OpenAI-compatible, so we use `llama-index-llms-openai-like`:

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from config import get_settings


_llm_instance: OpenAILike | None = None


def get_llm() -> OpenAILike:
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = OpenAILike(
            model=settings.deepseek_model,
            api_base=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            is_chat_model=True,
            timeout=120.0,
        )
        Settings.llm = _llm_instance
    return _llm_instance
```

- [ ] **Step 2: Create backend/services/embedding_service.py**

```python
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.core import Settings
from config import get_settings


_embedding_instance: DashScopeEmbedding | None = None


def get_embedding() -> DashScopeEmbedding:
    global _embedding_instance
    if _embedding_instance is None:
        settings = get_settings()
        _embedding_instance = DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
            api_key=settings.dashscope_api_key,
        )
        Settings.embed_model = _embedding_instance
    return _embedding_instance
```

- [ ] **Step 3: Commit**

```bash
git add backend/services/llm_service.py backend/services/embedding_service.py
git commit -m "feat: add DeepSeek LLM and DashScope embedding services"
```

---

## Task 4: Milvus Index Service

**Files:**
- Create: `backend/services/index_service.py`

- [ ] **Step 1: Create backend/services/index_service.py**

```python
from pymilvus import connections, utility, Collection
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from config import get_settings
from typing import Optional


_vector_store: Optional[MilvusVectorStore] = None
_vector_index: Optional[VectorStoreIndex] = None


def setup_callback_manager() -> None:
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager


def connect_milvus() -> None:
    settings = get_settings()
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=str(settings.milvus_port),
    )


def is_milvus_connected() -> bool:
    try:
        connections.connect(
            alias="check",
            host=get_settings().milvus_host,
            port=str(get_settings().milvus_port),
        )
        connections.disconnect("check")
        return True
    except Exception:
        return False


def get_vector_store() -> MilvusVectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=settings.milvus_collection,
            dim=1536,
            overwrite=False,
        )
    return _vector_store


def get_vector_index() -> VectorStoreIndex:
    global _vector_index
    if _vector_index is None:
        store = get_vector_store()
        _vector_index = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=Settings.embed_model,
        )
    return _vector_index


def reset_vector_index() -> None:
    global _vector_index
    _vector_index = None


def get_collection_count() -> int:
    settings = get_settings()
    try:
        if utility.has_collection(settings.milvus_collection):
            collection = Collection(settings.milvus_collection)
            collection.load()
            return collection.num_entities
        return 0
    except Exception:
        return 0


def delete_entities_by_file_id(file_id: str) -> int:
    settings = get_settings()
    try:
        if utility.has_collection(settings.milvus_collection):
            collection = Collection(settings.milvus_collection)
            collection.load()
            expr = f'file_id == "{file_id}"'
            result = collection.query(expr=expr, output_fields=["id"])
            if result:
                ids = [r["id"] for r in result]
                collection.delete(f"id in {ids}")
                collection.flush()
                return len(ids)
    except Exception:
        pass
    return 0
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/index_service.py
git commit -m "feat: add Milvus index service with connection and collection management"
```

---

## Task 5: Document Service (Upload + Ingestion Pipeline)

**Files:**
- Create: `backend/services/document_service.py`

This service handles file validation, parsing via LlamaIndex readers, metadata injection, and ingestion into Milvus via IngestionPipeline. The document registry is stored as a JSON file on disk.

- [ ] **Step 1: Create backend/services/document_service.py**

```python
import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

from config import get_settings
from services.index_service import get_vector_store, reset_vector_index

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".markdown"}

REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REGISTRY_FILE = os.path.join(REGISTRY_DIR, "documents.json")


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_registry(data: dict) -> None:
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def validate_file(filename: str, file_size: int) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"不支持的文件类型: {ext}。支持: {', '.join(ALLOWED_EXTENSIONS)}"
    settings = get_settings()
    if file_size > settings.max_file_size_bytes:
        return f"文件过大: {file_size / 1024 / 1024:.1f}MB，限制: {settings.max_file_size_mb}MB"
    return None


def _get_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {".txt": "txt", ".pdf": "pdf", ".md": "md", ".markdown": "md"}
    return mapping.get(ext, "unknown")


def _read_file_content(filepath: str, file_type: str) -> str:
    if file_type == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def ingest_file(filepath: str, filename: str) -> dict:
    settings = get_settings()
    file_id = str(uuid.uuid4())[:8]
    file_type = _get_file_type(filename)

    # Read and parse file content
    content = _read_file_content(filepath, file_type)

    # Create LlamaIndex Document with metadata
    doc = Document(
        text=content,
        metadata={
            "source_file": filename,
            "file_type": file_type,
            "file_id": file_id,
            "upload_time": datetime.now().isoformat(),
        },
    )

    # Build IngestionPipeline with SentenceSplitter
    node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    pipeline = IngestionPipeline(
        transformations=[
            node_parser,
            Settings.embed_model,
        ],
        vector_store=get_vector_store(),
    )

    # Run pipeline: parse -> embed -> store
    nodes = pipeline.run(documents=[doc])
    chunks_count = len(nodes)

    # Record in registry
    registry = _load_registry()
    registry[file_id] = {
        "file_id": file_id,
        "filename": filename,
        "file_type": file_type,
        "chunks_count": chunks_count,
        "upload_time": datetime.now().isoformat(),
        "filepath": filepath,
    }
    _save_registry(registry)

    # Reset cached index so next query picks up new data
    reset_vector_index()

    return {
        "file_id": file_id,
        "filename": filename,
        "chunks_count": chunks_count,
        "status": "success",
    }


def list_documents() -> list[dict]:
    registry = _load_registry()
    return list(registry.values())


def delete_document(file_id: str) -> bool:
    from services.index_service import delete_entities_by_file_id

    registry = _load_registry()
    if file_id not in registry:
        return False

    # Remove vectors from Milvus
    delete_entities_by_file_id(file_id)

    # Remove file from disk
    filepath = registry[file_id].get("filepath", "")
    if filepath and os.path.exists(filepath):
        os.remove(filepath)

    # Remove from registry
    del registry[file_id]
    _save_registry(registry)

    # Reset cached index
    reset_vector_index()

    return True


def get_document_count() -> int:
    return len(_load_registry())


def get_total_chunks() -> int:
    registry = _load_registry()
    return sum(doc.get("chunks_count", 0) for doc in registry.values())
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/document_service.py
git commit -m "feat: add document service with file parsing and IngestionPipeline"
```

---

## Task 6: Query Service (Core RAG Pipeline)

**Files:**
- Create: `backend/services/query_service.py`

This is the core of the application. It implements:
- Query rewriting (always on)
- 3 retrieval strategies: basic, hyde, window
- Optional reranking overlay
- ChatMemoryBuffer for multi-turn conversation
- System prompt engineering
- Prompt assembly for streaming

- [ ] **Step 1: Create backend/services/query_service.py**

```python
from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from config import get_settings
from services.index_service import get_vector_index
from services.llm_service import get_llm
from services.document_service import get_document_count

# System prompt template
SYSTEM_PROMPT = """你是一个专业的企业知识库助手。请基于提供的上下文文档回答用户问题。

规则：
1. 仅基于提供的上下文文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"文档中未找到相关信息"
3. 回答中引用来源时，使用 [来源: 文件名] 格式标注
4. 使用Markdown格式组织回答，合理使用标题、列表、代码块

上下文文档：
{context}"""

# In-memory chat memory (single session)
_chat_memory: ChatMemoryBuffer | None = None


def get_chat_memory() -> ChatMemoryBuffer:
    global _chat_memory
    if _chat_memory is None:
        settings = get_settings()
        _chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=settings.chat_token_limit,
            tokenizer=Settings.tokenizer,
        )
    return _chat_memory


def reset_chat_memory() -> None:
    global _chat_memory
    _chat_memory = None


def rewrite_query(question: str) -> str:
    """Rewrite user query into a more retrieval-friendly form."""
    llm = get_llm()
    rewrite_prompt = PromptTemplate(
        "你是一个查询重写助手。请将用户的口语化提问重写为更适合知识库检索的形式。\n"
        "要求：保持原始意图，使用更正式和完整的表述，不要回答问题。\n"
        "原始问题：{question}\n"
        "重写后的问题："
    )
    response = llm.predict(rewrite_prompt, question=question)
    return response.strip()


def retrieve_basic(query: str) -> list[NodeWithScore]:
    """Basic vector similarity retrieval."""
    settings = get_settings()
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=settings.top_k)
    return retriever.retrieve(query)


def retrieve_hyde(query: str) -> list[NodeWithScore]:
    """HyDE: generate hypothetical document, then retrieve with its embedding."""
    settings = get_settings()
    llm = get_llm()
    index = get_vector_index()

    # Generate hypothetical answer
    hyde_prompt = PromptTemplate(
        "请基于你的知识，简要回答以下问题。如果你不确定，请给出合理的猜测：\n{query}"
    )
    hypothetical_answer = llm.predict(hyde_prompt, query=query)

    # Retrieve using the hypothetical answer as the query
    retriever = index.as_retriever(similarity_top_k=settings.top_k)
    return retriever.retrieve(hypothetical_answer)


def retrieve_window(query: str) -> list[NodeWithScore]:
    """Sentence Window retrieval: retrieve small chunks, expand with surrounding context."""
    settings = get_settings()
    index = get_vector_index()

    # Create query engine with MetadataReplacementPostProcessor
    query_engine = index.as_query_engine(
        similarity_top_k=settings.top_k,
        node_postprocessors=[
            MetadataReplacementPostProcessor(
                target_metadata_key="window",
            ),
        ],
    )
    response = query_engine.retrieve(query)
    return response


def rerank_nodes(
    nodes: list[NodeWithScore], query: str
) -> list[NodeWithScore]:
    """Apply CrossEncoder reranking to refine retrieval results."""
    settings = get_settings()
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=settings.top_n,
    )
    return reranker.postprocess_nodes(nodes, query_str=query)


def retrieve(
    question: str, strategy: str = "basic", use_rerank: bool = True
) -> list[NodeWithScore]:
    """Full retrieval pipeline: rewrite -> strategy -> optional rerank."""
    # Step 1: Query rewriting
    rewritten = rewrite_query(question)

    # Step 2: Strategy-based retrieval
    if strategy == "hyde":
        nodes = retrieve_hyde(rewritten)
    elif strategy == "window":
        nodes = retrieve_window(rewritten)
    else:
        nodes = retrieve_basic(rewritten)

    # Step 3: Optional reranking
    if use_rerank and nodes:
        nodes = rerank_nodes(nodes, rewritten)

    return nodes


def build_prompt(question: str, nodes: list[NodeWithScore]) -> str:
    """Assemble the full prompt with context, memory, and user question."""
    memory = get_chat_memory()

    # Build context string from retrieved nodes
    context_parts = []
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source_file", "未知来源")
        text = node.node.get_content()
        context_parts.append(f"[文档 {i} | 来源: {source}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # Build chat history string
    history_msgs = memory.get()
    history_text = ""
    if history_msgs:
        parts = []
        for msg in history_msgs:
            role = "用户" if msg.role == "user" else "助手"
            parts.append(f"{role}: {msg.content}")
        history_text = "\n".join(parts)

    # Assemble full prompt
    system = SYSTEM_PROMPT.format(context=context)
    sections = [system]
    if history_text:
        sections.append(f"会话历史：\n{history_text}")
    sections.append(f"用户问题：{question}")
    return "\n\n".join(sections)


def get_sources(nodes: list[NodeWithScore]) -> list[dict]:
    """Extract source info from retrieved nodes for frontend display."""
    sources = []
    for node in nodes:
        sources.append({
            "source_file": node.node.metadata.get("source_file", "未知"),
            "score": round(node.score, 4) if node.score else 0.0,
        })
    return sources


def chat(
    question: str, strategy: str = "basic", use_rerank: bool = True
) -> tuple:
    """
    Execute full RAG pipeline.
    Returns: (prompt: str, sources: list[dict], has_knowledge: bool)
    """
    if get_document_count() == 0:
        return (
            "当前知识库为空，请先上传文档。",
            [],
            False,
        )

    # Retrieve relevant nodes
    nodes = retrieve(question, strategy, use_rerank)

    # Build prompt
    prompt = build_prompt(question, nodes)

    # Extract sources
    sources = get_sources(nodes)

    # Add user message to memory
    memory = get_chat_memory()
    memory.put(question, "user")

    return prompt, sources, True
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/query_service.py
git commit -m "feat: add query service with rewrite, 3 strategies, rerank, memory"
```

---

## Task 7: FastAPI Routes & App Entry

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Create backend/main.py**

```python
import os
import shutil
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models.schemas import (
    ChatRequest, UploadResponse, DeleteResponse,
    AppConfig, AppStats, DocumentInfo,
)
from services.llm_service import get_llm
from services.embedding_service import get_embedding
from services.index_service import (
    setup_callback_manager, connect_milvus, is_milvus_connected,
    get_collection_count,
)
from services.document_service import (
    validate_file, ingest_file, list_documents,
    delete_document, get_document_count, get_total_chunks,
)
from services.query_service import chat, get_llm as get_query_llm, get_chat_memory

app = FastAPI(title="企业级RAG知识库助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.on_event("startup")
async def startup():
    setup_callback_manager()
    get_embedding()
    get_llm()
    try:
        connect_milvus()
    except Exception as e:
        print(f"警告: Milvus连接失败 — {e}")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    settings = get_settings()
    content = await file.read()

    # Validate
    error = validate_file(file.filename, len(content))
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Save to uploads directory
    upload_dir = os.path.join(os.path.dirname(__file__), settings.upload_dir)
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(content)

    # Ingest into Milvus
    try:
        result = ingest_file(filepath, file.filename)
        return UploadResponse(**result)
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"文件入库失败: {str(e)}")


@app.get("/api/documents", response_model=list[DocumentInfo])
async def get_documents():
    docs = list_documents()
    return [DocumentInfo(**doc) for doc in docs]


@app.delete("/api/documents/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: str):
    success = delete_document(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")
    return DeleteResponse(status="ok", file_id=file_id)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Run RAG pipeline
    prompt, sources, has_knowledge = chat(
        question=request.question,
        strategy=request.strategy,
        use_rerank=request.use_rerank,
    )

    if not has_knowledge:
        async def empty_stream():
            msg = "当前知识库为空，请先上传文档后再提问。"
            for char in msg:
                yield f"data: {__import__('json').dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {__import__('json').dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    async def event_stream():
        import json as json_module

        # 1. Thinking event
        yield f"data: {json_module.dumps({'type': 'thinking', 'content': '正在检索知识库...'}, ensure_ascii=False)}\n\n"

        # 2. Sources event
        yield f"data: {json_module.dumps({'type': 'sources', 'content': sources}, ensure_ascii=False)}\n\n"

        # 3. Stream LLM response
        try:
            llm = get_query_llm()
            response = llm.stream_complete(prompt)
            full_response = ""
            for chunk in response:
                delta = chunk.delta
                if delta:
                    full_response += delta
                    event = json_module.dumps(
                        {"type": "token", "content": delta},
                        ensure_ascii=False,
                    )
                    yield f"data: {event}\n\n"

            # Save assistant response to memory
            memory = get_chat_memory()
            memory.put(full_response, "assistant")

        except Exception as e:
            error_event = json_module.dumps(
                {"type": "error", "content": f"生成回答失败: {str(e)}"},
                ensure_ascii=False,
            )
            yield f"data: {error_event}\n\n"

        # 4. Done event
        yield f"data: {json_module.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/config", response_model=AppConfig)
async def get_config():
    settings = get_settings()
    return AppConfig(
        strategies=["basic", "hyde", "window"],
        default_strategy=settings.default_strategy,
        rerank_enabled=settings.default_use_rerank,
        llm_model=settings.deepseek_model,
        embedding_model="text-embedding-v2",
    )


@app.get("/api/stats", response_model=AppStats)
async def get_stats():
    return AppStats(
        documents=get_document_count(),
        chunks=get_total_chunks(),
        milvus_connected=is_milvus_connected(),
    )


# Mount static files LAST (so /api/* routes take priority)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
```

- [ ] **Step 2: Commit**

```bash
git add backend/main.py
git commit -m "feat: add FastAPI routes with upload, chat SSE, documents CRUD"
```

---

## Task 8: Frontend — HTML Structure

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Create frontend/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>企业级RAG知识库助手</title>
    <link rel="stylesheet" href="/css/style.css">
    <!-- Markdown rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Code highlighting -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
</head>
<body>
    <div class="app">
        <!-- Header -->
        <header class="header">
            <h1>🏢 企业级RAG知识库助手</h1>
            <div class="header-stats">
                <span id="stat-docs">📄 0 文档</span>
                <span id="stat-chunks">📦 0 分块</span>
                <span id="stat-milvus" class="status-dot disconnected">Milvus</span>
            </div>
        </header>

        <div class="main">
            <!-- Left Sidebar -->
            <aside class="sidebar">
                <!-- Upload Section -->
                <div class="sidebar-section">
                    <h2>📁 知识库管理</h2>
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">📤</div>
                        <p>拖拽文件到此处或点击上传</p>
                        <p class="upload-hint">支持 .txt / .pdf / .md</p>
                        <input type="file" id="fileInput" accept=".txt,.pdf,.md,.markdown" hidden>
                    </div>
                    <div id="uploadProgress" class="upload-progress hidden">
                        <div class="progress-bar"><div class="progress-fill"></div></div>
                        <span id="uploadStatus">上传中...</span>
                    </div>
                </div>

                <!-- Document List -->
                <div class="sidebar-section">
                    <h3>📋 文件列表</h3>
                    <div id="docList" class="doc-list">
                        <p class="empty-hint">暂无文档</p>
                    </div>
                    <button id="deleteSelectedBtn" class="btn btn-danger btn-sm hidden">🗑️ 删除选中</button>
                </div>

                <!-- Strategy Selection -->
                <div class="sidebar-section">
                    <h3>🔍 检索策略</h3>
                    <div class="strategy-group">
                        <label class="radio-label" title="直接向量相似度检索，速度最快">
                            <input type="radio" name="strategy" value="basic" checked>
                            <span>基础检索</span>
                        </label>
                        <label class="radio-label" title="先生成假设答案文档，再用其嵌入检索，提升语义匹配">
                            <input type="radio" name="strategy" value="hyde">
                            <span>HyDE 假设文档</span>
                        </label>
                        <label class="radio-label" title="检索小块文本后扩展上下文窗口，兼顾精确和完整">
                            <input type="radio" name="strategy" value="window">
                            <span>句子窗口</span>
                        </label>
                    </div>
                    <label class="checkbox-label" title="交叉编码器精排，可叠加在任意检索策略上">
                        <input type="checkbox" id="useRerank" checked>
                        <span>☑ Re-ranking 重排序</span>
                    </label>
                </div>
            </aside>

            <!-- Chat Area -->
            <main class="chat-area">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            <p>你好！我是企业知识库助手。请先在左侧上传文档，然后向我提问。</p>
                        </div>
                    </div>
                </div>

                <!-- Input Area -->
                <div class="chat-input-area">
                    <div class="input-row">
                        <textarea
                            id="questionInput"
                            placeholder="输入你的问题..."
                            rows="1"
                        ></textarea>
                        <button id="sendBtn" class="btn btn-primary" title="发送 (Enter)">
                            ▶ 发送
                        </button>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script src="/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add frontend HTML structure with sidebar and chat layout"
```

---

## Task 9: Frontend — CSS Styles

**Files:**
- Create: `frontend/css/style.css`

- [ ] **Step 1: Create frontend/css/style.css**

```css
/* ===== CSS Variables ===== */
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-sidebar: #0f3460;
    --bg-input: #1a1a2e;
    --text-primary: #e8e8e8;
    --text-secondary: #a0a0b0;
    --accent: #4fc3f7;
    --accent-hover: #29b6f6;
    --danger: #ef5350;
    --success: #66bb6a;
    --border: #2a2a4a;
    --msg-user-bg: #1b5e20;
    --msg-assistant-bg: #1a237e;
    --radius: 8px;
    --shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* ===== Reset & Base ===== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
}

/* ===== App Layout ===== */
.app {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}

.header h1 {
    font-size: 18px;
    font-weight: 600;
}

.header-stats {
    display: flex;
    gap: 16px;
    font-size: 13px;
    color: var(--text-secondary);
}

.status-dot {
    display: flex;
    align-items: center;
    gap: 4px;
}

.status-dot::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--danger);
}

.status-dot.connected::before {
    background: var(--success);
}

.main {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* ===== Sidebar ===== */
.sidebar {
    width: 280px;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
    flex-shrink: 0;
}

.sidebar-section h2 {
    font-size: 15px;
    margin-bottom: 10px;
    color: var(--accent);
}

.sidebar-section h3 {
    font-size: 13px;
    margin-bottom: 8px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ===== Upload Area ===== */
.upload-area {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
}

.upload-area:hover,
.upload-area.dragover {
    border-color: var(--accent);
    background: rgba(79, 195, 247, 0.05);
}

.upload-icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.upload-area p {
    font-size: 13px;
    color: var(--text-secondary);
}

.upload-hint {
    font-size: 11px !important;
    margin-top: 4px;
}

.upload-progress {
    margin-top: 8px;
}

.progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.3s;
    animation: progress-anim 1.5s ease-in-out infinite;
}

@keyframes progress-anim {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
}

#uploadStatus {
    font-size: 12px;
    color: var(--text-secondary);
}

/* ===== Document List ===== */
.doc-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 200px;
    overflow-y: auto;
}

.doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s;
}

.doc-item:hover {
    background: rgba(255, 255, 255, 0.08);
}

.doc-item input[type="checkbox"] {
    accent-color: var(--accent);
}

.doc-icon {
    font-size: 16px;
}

.doc-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.doc-chunks {
    font-size: 11px;
    color: var(--text-secondary);
}

.empty-hint {
    font-size: 13px;
    color: var(--text-secondary);
    text-align: center;
    padding: 12px;
}

/* ===== Strategy Selection ===== */
.strategy-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 10px;
}

.radio-label,
.checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    cursor: pointer;
    padding: 4px 0;
}

.radio-label input,
.checkbox-label input {
    accent-color: var(--accent);
}

/* ===== Chat Area ===== */
.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.message {
    display: flex;
    gap: 10px;
    max-width: 85%;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.message-content {
    background: var(--msg-assistant-bg);
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 14px;
    word-break: break-word;
}

.message.user .message-content {
    background: var(--msg-user-bg);
}

.message-content p {
    margin-bottom: 8px;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-content pre {
    background: rgba(0, 0, 0, 0.3);
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 8px 0;
}

.message-content code {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 13px;
}

.message-content ul,
.message-content ol {
    padding-left: 20px;
    margin: 4px 0;
}

/* Source badges */
.sources-container {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.source-badge {
    display: inline-block;
    background: rgba(79, 195, 247, 0.15);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    margin: 2px 4px 2px 0;
}

/* Thinking indicator */
.thinking {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 13px;
    padding: 4px 0;
}

.thinking-dots span {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
}

/* ===== Input Area ===== */
.chat-input-area {
    padding: 16px 20px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
}

.input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

textarea {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 14px;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    resize: none;
    max-height: 120px;
    line-height: 1.5;
    transition: border-color 0.2s;
}

textarea:focus {
    outline: none;
    border-color: var(--accent);
}

textarea::placeholder {
    color: var(--text-secondary);
}

/* ===== Buttons ===== */
.btn {
    border: none;
    border-radius: var(--radius);
    padding: 10px 18px;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    font-weight: 500;
}

.btn:active {
    transform: scale(0.97);
}

.btn-primary {
    background: var(--accent);
    color: #000;
}

.btn-primary:hover {
    background: var(--accent-hover);
}

.btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.btn-danger {
    background: var(--danger);
    color: #fff;
}

.btn-danger:hover {
    background: #d32f2f;
}

.btn-sm {
    padding: 6px 12px;
    font-size: 12px;
    margin-top: 8px;
    width: 100%;
}

.hidden {
    display: none !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #444;
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
    .sidebar {
        width: 220px;
    }
    .message {
        max-width: 95%;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/css/style.css
git commit -m "feat: add frontend CSS with dark theme and responsive layout"
```

---

## Task 10: Frontend — JavaScript Logic

**Files:**
- Create: `frontend/js/app.js`

- [ ] **Step 1: Create frontend/js/app.js**

```javascript
// ===== State =====
const state = {
    isStreaming: false,
    currentStrategy: 'basic',
    useRerank: true,
};

// ===== DOM Elements =====
const $ = (sel) => document.querySelector(sel);
const chatMessages = $('#chatMessages');
const questionInput = $('#questionInput');
const sendBtn = $('#sendBtn');
const fileInput = $('#fileInput');
const uploadArea = $('#uploadArea');
const uploadProgress = $('#uploadProgress');
const uploadStatus = $('#uploadStatus');
const docList = $('#docList');
const deleteSelectedBtn = $('#deleteSelectedBtn');
const useRerankCheckbox = $('#useRerank');

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    loadStats();
    loadConfig();
    setupEventListeners();
});

// ===== Event Listeners =====
function setupEventListeners() {
    // Send message
    sendBtn.addEventListener('click', sendMessage);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    });

    // File upload
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });

    // Strategy selection
    document.querySelectorAll('input[name="strategy"]').forEach((radio) => {
        radio.addEventListener('change', (e) => {
            state.currentStrategy = e.target.value;
        });
    });

    // Rerank toggle
    useRerankCheckbox.addEventListener('change', (e) => {
        state.useRerank = e.target.checked;
    });

    // Delete selected
    deleteSelectedBtn.addEventListener('click', deleteSelectedDocs);
}

// ===== API: Upload File =====
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.classList.remove('hidden');
    uploadStatus.textContent = `正在上传: ${file.name}`;

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || '上传失败');
        }

        uploadStatus.textContent = `✅ ${file.name} — ${data.chunks_count} 个分块`;
        setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
        loadDocuments();
        loadStats();
    } catch (err) {
        uploadStatus.textContent = `❌ ${err.message}`;
        setTimeout(() => uploadProgress.classList.add('hidden'), 5000);
    }

    fileInput.value = '';
}

// ===== API: Load Documents =====
async function loadDocuments() {
    try {
        const res = await fetch('/api/documents');
        const docs = await res.json();
        renderDocList(docs);
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

function renderDocList(docs) {
    if (docs.length === 0) {
        docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
        deleteSelectedBtn.classList.add('hidden');
        return;
    }

    const icons = { txt: '📝', pdf: '📕', md: '📋' };
    docList.innerHTML = docs.map((doc) => `
        <div class="doc-item">
            <input type="checkbox" data-id="${doc.file_id}" class="doc-checkbox">
            <span class="doc-icon">${icons[doc.file_type] || '📄'}</span>
            <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
            <span class="doc-chunks">${doc.chunks_count}块</span>
        </div>
    `).join('');

    deleteSelectedBtn.classList.remove('hidden');

    // Show/hide delete button based on checkbox state
    docList.querySelectorAll('.doc-checkbox').forEach((cb) => {
        cb.addEventListener('change', () => {
            const anyChecked = docList.querySelectorAll('.doc-checkbox:checked').length > 0;
            deleteSelectedBtn.classList.toggle('hidden', !anyChecked);
        });
    });
}

// ===== API: Delete Documents =====
async function deleteSelectedDocs() {
    const checked = docList.querySelectorAll('.doc-checkbox:checked');
    if (checked.length === 0) return;

    for (const cb of checked) {
        const fileId = cb.dataset.id;
        await fetch(`/api/documents/${fileId}`, { method: 'DELETE' });
    }

    loadDocuments();
    loadStats();
}

// ===== API: Load Stats =====
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        $('#stat-docs').textContent = `📄 ${stats.documents} 文档`;
        $('#stat-chunks').textContent = `📦 ${stats.chunks} 分块`;
        const milvusDot = $('#stat-milvus');
        milvusDot.textContent = 'Milvus';
        milvusDot.classList.toggle('connected', stats.milvus_connected);
        milvusDot.classList.toggle('disconnected', !stats.milvus_connected);
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

// ===== API: Load Config =====
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        // Set default strategy
        const radio = document.querySelector(`input[name="strategy"][value="${config.default_strategy}"]`);
        if (radio) {
            radio.checked = true;
            state.currentStrategy = config.default_strategy;
        }
        useRerankCheckbox.checked = config.rerank_enabled;
        state.useRerank = config.rerank_enabled;
    } catch (err) {
        console.error('Failed to load config:', err);
    }
}

// ===== Chat: Send Message =====
async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || state.isStreaming) return;

    state.isStreaming = true;
    sendBtn.disabled = true;

    // Add user message to UI
    appendMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';

    // Create assistant message placeholder
    const assistantEl = appendMessage('assistant', '');
    const contentEl = assistantEl.querySelector('.message-content');

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                strategy: state.currentStrategy,
                use_rerank: state.useRerank,
            }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let sources = [];
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const event = JSON.parse(jsonStr);
                    handleSSEEvent(event, contentEl, { fullText, sources });

                    if (event.type === 'token') {
                        fullText += event.content;
                        renderMarkdown(contentEl, fullText, sources);
                    } else if (event.type === 'sources') {
                        sources = event.content;
                    }
                } catch (e) {
                    // Skip malformed JSON
                }
            }
        }

        // Final render with sources
        renderMarkdown(contentEl, fullText, sources);
    } catch (err) {
        contentEl.innerHTML = `<p style="color: var(--danger);">❌ 请求失败: ${err.message}</p>`;
    }

    state.isStreaming = false;
    sendBtn.disabled = false;
    questionInput.focus();
}

// ===== SSE Event Handler =====
function handleSSEEvent(event, contentEl, ctx) {
    switch (event.type) {
        case 'thinking':
            contentEl.innerHTML = `
                <div class="thinking">
                    <div class="thinking-dots"><span></span><span></span><span></span></div>
                    <span>${event.content}</span>
                </div>`;
            break;
        case 'error':
            contentEl.innerHTML = `<p style="color: var(--danger);">⚠️ ${event.content}</p>`;
            break;
        case 'done':
            // Handled after stream ends
            break;
    }
}

// ===== Render Markdown =====
function renderMarkdown(el, text, sources) {
    if (!text && sources.length === 0) return;

    // Configure marked + highlight.js
    marked.setOptions({
        highlight: (code, lang) => {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
    });

    let html = text ? marked.parse(text) : '';

    // Append source badges
    if (sources.length > 0) {
        const uniqueSources = [...new Map(sources.map(s => [s.source_file, s])).values()];
        html += `<div class="sources-container"><strong style="font-size:12px;">📎 引用来源:</strong><br>`;
        html += uniqueSources.map(s =>
            `<span class="source-badge">${s.source_file} (${(s.score * 100).toFixed(0)}%)</span>`
        ).join('');
        html += '</div>';
    }

    el.innerHTML = html;
    scrollToBottom();
}

// ===== UI Helpers =====
function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const avatar = role === 'user' ? '👤' : '🤖';
    const rendered = content ? marked.parse(content) : '';
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${rendered}</div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/app.js
git commit -m "feat: add frontend JS with SSE chat, file upload, strategy switching"
```

---

## Task 11: Smoke Test & Final Commit

**Files:**
- Modify: `.gitignore` (if needed)

- [ ] **Step 1: Start Milvus (if not running)**

Run: `docker compose up -d`
Expected: All 3 containers running. `docker compose ps` shows healthy status.

- [ ] **Step 2: Create .env from template**

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `DEEPSEEK_API_KEY` — your DeepSeek API key
- `DASHSCOPE_API_KEY` — your DashScope/百炼 API key

- [ ] **Step 3: Start the backend**

Run:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Expected: Server starts without errors. Console shows "Uvicorn running on http://0.0.0.0:8000".

- [ ] **Step 4: Verify API endpoints**

Test config endpoint:
```bash
curl http://localhost:8000/api/config
```
Expected JSON: `{"strategies":["basic","hyde","window"],"default_strategy":"basic","rerank_enabled":true,"llm_model":"deepseek-chat","embedding_model":"text-embedding-v2"}`

Test stats endpoint:
```bash
curl http://localhost:8000/api/stats
```
Expected JSON: `{"documents":0,"chunks":0,"milvus_connected":true}`

Test documents list:
```bash
curl http://localhost:8000/api/documents
```
Expected: `[]`

- [ ] **Step 5: Upload a test document**

Create a test file:
```bash
echo "LlamaIndex是一个用于构建RAG应用的开源框架。它提供了文档加载、文本分块、向量索引和检索等核心功能。" > backend/uploads/test.txt
```

Upload via API:
```bash
curl -X POST http://localhost:8000/api/upload -F "file=@backend/uploads/test.txt"
```
Expected: `{"file_id":"...","filename":"test.txt","chunks_count":1,"status":"success"}`

- [ ] **Step 6: Test chat endpoint**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"LlamaIndex是什么?","strategy":"basic","use_rerank":false}'
```
Expected: SSE stream with `thinking`, `sources`, `token` events, and a `done` event. The answer should reference the uploaded test document.

- [ ] **Step 7: Open frontend in browser**

Open: `http://localhost:8000`
Expected: Chat UI loads with sidebar, strategy selector, and welcome message. Uploaded document appears in file list.

- [ ] **Step 8: Test full flow in browser**

1. Upload a PDF or markdown file via drag-and-drop
2. Select "HyDE" strategy with Re-ranking enabled
3. Ask a question about the uploaded document
4. Verify streaming response with source citations

- [ ] **Step 9: Final commit**

```bash
git add -A
git commit -m "feat: complete enterprise RAG application with LlamaIndex"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task(s) | Status |
|---|---|---|
| File upload (txt/pdf/md) | Task 5, Task 7 | ✅ |
| 3 retrieval strategies + rerank toggle | Task 6 | ✅ |
| LlamaIndex Workflows / Pipeline | Task 5 (IngestionPipeline), Task 6 (query pipeline) | ✅ |
| Router Query Engine | Task 6 (strategy dispatch) | ✅ |
| ChatMemoryBuffer multi-turn | Task 6 | ✅ |
| SSE streaming | Task 7, Task 10 | ✅ |
| Callback Manager observability | Task 4 | ✅ |
| DeepSeek LLM | Task 3 | ✅ |
| DashScope embedding | Task 3 | ✅ |
| Milvus Standalone | Task 1, Task 4 | ✅ |
| Frontend (HTML/CSS/JS) | Task 8, Task 9, Task 10 | ✅ |
| Error handling | Task 5 (validation), Task 7 (HTTP errors), Task 10 (UI errors) | ✅ |
| System Prompt engineering | Task 6 | ✅ |
| Evaluation (预留) | Not in MVP per spec | ✅ (excluded by design) |

### Placeholder Scan
- No TBD/TODO found
- All code blocks contain complete implementations
- All commands have expected outputs

### Type Consistency
- `ChatRequest` fields (`question`, `strategy`, `use_rerank`) match between schemas.py, main.py, and app.js
- `UploadResponse` fields (`file_id`, `filename`, `chunks_count`, `status`) match between schemas.py, document_service.py, and main.py
- `DocumentInfo` fields match between schemas.py, document_service.py registry, and main.py
- `AppConfig` fields match between schemas.py, main.py, and loadConfig() in app.js
- SSE event types (`thinking`, `sources`, `token`, `done`, `error`) match between main.py and app.js
- Strategy values (`basic`, `hyde`, `window`) match between schemas.py, main.py, query_service.py, and app.js
