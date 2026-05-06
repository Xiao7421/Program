from abc import abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
import os

from utils.config_handler import rag_conf


class BaseModelFactory:
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory:
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            streaming=True,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        )


class EmbeddingModelFactory:
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        )


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingModelFactory().generator()
