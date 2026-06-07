import os
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
)
from services.document_service import (
    validate_file, ingest_file, list_documents,
    delete_document, get_document_count, get_total_chunks,
)
from services.query_service import chat, get_chat_memory
from llama_index.core.base.llms.types import ChatMessage, MessageRole

app = FastAPI(title="企业级RAG知识库助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = str(PROJECT_ROOT / "frontend")


@app.on_event("startup")
async def startup():
    import httpx

    settings = get_settings()

    setup_callback_manager()
    get_embedding()
    get_llm()

    # Milvus health check
    try:
        connect_milvus()
        print(f"✅ Milvus 连接成功 — {settings.milvus_host}:{settings.milvus_port}")
    except Exception as e:
        print(f"❌ Milvus 连接失败 — {settings.milvus_host}:{settings.milvus_port}: {e}")

    # Reranker health check
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(settings.reranker_url.replace("/rerank", "/health"))
            if resp.is_success:
                print(f"✅ Reranker 连接成功 — {settings.reranker_url}")
            else:
                print(f"⚠️  Reranker 健康检查异常 — {settings.reranker_url} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"❌ Reranker 连接失败 — {settings.reranker_url}: {e}")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    settings = get_settings()
    content = await file.read()

    # Validate
    error = validate_file(file.filename, len(content))
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Save to uploads directory
    upload_dir = str(BACKEND_DIR / "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(content)

    # Ingest into Milvus
    try:
        result = ingest_file(filepath, file.filename)
        return UploadResponse(**result)
    except Exception as e:
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
            import json as json_module
            msg = "当前知识库为空，请先上传文档后再提问。"
            for char in msg:
                yield f"data: {json_module.dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json_module.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    async def event_stream():
        import json as json_module

        # 1. Thinking event
        yield f"data: {json_module.dumps({'type': 'thinking', 'content': '正在检索知识库...'}, ensure_ascii=False)}\n\n"

        # 2. Sources event
        yield f"data: {json_module.dumps({'type': 'sources', 'content': sources}, ensure_ascii=False)}\n\n"

        # 3. Stream LLM response (async to avoid blocking event loop)
        try:
            llm = get_llm()
            response = await llm.astream_complete(prompt)
            full_response = ""
            async for chunk in response:
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
            memory.put(ChatMessage(content=full_response, role=MessageRole.ASSISTANT))

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
        embedding_model="text-embedding-v4",
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
