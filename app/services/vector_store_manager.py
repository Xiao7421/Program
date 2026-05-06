"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

import time
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from pymilvus import Collection

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service
from app.core.milvus_client import milvus_manager


# 统一使用 biz collection
COLLECTION_NAME = "biz"
# 相似度搜索默认返回数量
DEFAULT_TOP_K = 3


class _VectorStoreWrapper:
    """LangChain Milvus 兼容包装器

    使用已连接的 milvus_manager 和 embeddings 服务，
    避免 langchain_milvus.Milvus 的连接别名问题。
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedding_function = vector_embedding_service

    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K, **kwargs) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            # 1. 向量化查询文本
            query_vector = self.embedding_function.embed_query(query)

            # 2. 获取 collection 并搜索
            collection: Collection = milvus_manager.get_collection()
            collection.load()

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["content", "metadata"],
            )

            # 3. 将结果转为 Document
            docs: List[Document] = []
            for hits in results:
                for hit in hits:
                    content = hit.entity.get("content") or ""
                    metadata = hit.entity.get("metadata") or {}
                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)

            logger.debug(f"相似度搜索完成: query='{query[:50]}...', 结果数={len(docs)}")
            return docs

        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs) -> List[str]:
        """
        批量添加文档到向量存储

        Args:
            documents: 文档列表
            ids: 文档 ID 列表（可选，自动生成）

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            start_time = time.time()
            n = len(documents)

            # 1. 生成 IDs
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]

            # 2. 批量向量化
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_function.embed_documents(texts)

            # 3. 准备插入数据
            collection: Collection = milvus_manager.get_collection()
            collection.load()

            entities = []
            for i, doc in enumerate(documents):
                entity = {
                    "id": ids[i],
                    "vector": embeddings[i],
                    "content": doc.page_content,
                    "metadata": doc.metadata or {},
                }
                entities.append(entity)

            # 4. 批量插入
            insert_result = collection.insert(entities)
            collection.flush()

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {n} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed / n:.2f}秒/个"
            )
            return ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store: Optional[_VectorStoreWrapper] = None
        self.collection_name = COLLECTION_NAME
        self._initialized = False

    def ensure_initialized(self):
        """确保向量存储已初始化（延迟初始化，避免导入时连接 Milvus）"""
        if not self._initialized:
            self._initialize_vector_store()
            self._initialized = True

    def _initialize_vector_store(self):
        """初始化向量存储"""
        try:
            # 确保 collection 已存在
            if not milvus_manager.get_collection():
                raise RuntimeError("Milvus collection 未初始化")

            self.vector_store = _VectorStoreWrapper(
                collection_name=self.collection_name,
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        if not self.vector_store:
            raise RuntimeError("VectorStore 未初始化，请先调用 ensure_initialized()")
        return self.vector_store.add_documents(documents)

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self):
        """
        获取 VectorStore 实例

        Returns:
            _VectorStoreWrapper: 向量存储包装器
        """
        return self.vector_store

    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        if not self.vector_store:
            logger.error("VectorStore 未初始化")
            return []
        return self.vector_store.similarity_search(query, k=k)


# 全局单例（延迟初始化，需调用 ensure_initialized()）
vector_store_manager = VectorStoreManager()
