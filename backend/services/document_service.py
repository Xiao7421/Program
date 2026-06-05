import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.ingestion import IngestionPipeline

from config import get_settings
from services.index_service import get_vector_store, reset_vector_index

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".markdown"}

BACKEND_DIR = Path(__file__).resolve().parent.parent
REGISTRY_DIR = str(BACKEND_DIR / "data")
REGISTRY_FILE = str(BACKEND_DIR / "data" / "documents.json")


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

    # Build IngestionPipeline with SentenceWindowNodeParser
    # This creates small chunks for precise retrieval, with a "window" metadata
    # field containing surrounding context for the Sentence Window strategy
    node_parser = SentenceWindowNodeParser(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
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
