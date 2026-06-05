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
