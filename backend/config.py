from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os

# Find project root (parent of backend/) and locate .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = str(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    # DeepSeek LLM
    deepseek_api_key: str = "sk-c7bf69b1ca51407fb31063e6961fc1c7"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # DashScope Embedding
    dashscope_api_key: str = "sk-ed99b70131734eba9202ecb8fb004edb"

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

    # Reranker (Docker service, e.g. TEI / Infinity)
    reranker_url: str = "http://localhost:8081/rerank"

    # File Upload
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # Embedding
    embedding_dim: int = 1536

    # Defaults
    default_strategy: str = "basic"
    default_use_rerank: bool = False

    model_config = {
        "env_file": ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
