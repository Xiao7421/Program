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
