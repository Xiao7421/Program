from pymilvus import MilvusClient
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from config import get_settings
from typing import Optional


_vector_store: Optional[MilvusVectorStore] = None
_vector_index: Optional[VectorStoreIndex] = None


def _get_milvus_uri() -> str:
    s = get_settings()
    return f"http://{s.milvus_host}:{s.milvus_port}"


def setup_callback_manager() -> None:
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager


def connect_milvus() -> None:
    # Verify Milvus is reachable by instantiating a client
    client = MilvusClient(uri=_get_milvus_uri())
    client.list_collections()


def is_milvus_connected() -> bool:
    try:
        client = MilvusClient(uri=_get_milvus_uri())
        client.list_collections()
        return True
    except Exception:
        return False


def get_vector_store() -> MilvusVectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = MilvusVectorStore(
            uri=_get_milvus_uri(),
            collection_name=settings.milvus_collection,
            dim=settings.embedding_dim,
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
        client = MilvusClient(uri=_get_milvus_uri())
        stats = client.get_collection_stats(settings.milvus_collection)
        return stats.get("row_count", 0)
    except Exception:
        return 0


def delete_entities_by_file_id(file_id: str) -> int:
    settings = get_settings()
    try:
        client = MilvusClient(uri=_get_milvus_uri())
        expr = f'file_id == "{file_id}"'
        result = client.query(
            collection_name=settings.milvus_collection,
            filter=expr,
            output_fields=["id"],
        )
        if result:
            ids = [r["id"] for r in result]
            client.delete(
                collection_name=settings.milvus_collection,
                filter=f"id in {ids}",
            )
            return len(ids)
    except Exception:
        pass
    return 0
