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
