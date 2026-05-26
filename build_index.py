"""构建扫地机器人知识库向量索引 (ChromaDB)"""
import os
import logging
import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index():
    """加载知识库文档，构建并持久化 ChromaDB 向量索引"""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请在 .env 文件中设置 DASHSCOPE_API_KEY")
        return

    # 初始化 embedding 模型
    logger.info("初始化 embedding 模型...")
    embed_model = DashScopeEmbedding(
        model_name="text-embedding-v4",
        api_key=api_key,
        embed_batch_size=10,
    )
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    # 加载文档
    logger.info("加载知识库文档...")
    try:
        documents = SimpleDirectoryReader("knowledge_base").load_data()
    except ValueError:
        logger.warning("知识库为空 (knowledge_base/ 中没有文件)，请先上传文档")
        return
    logger.info(f"加载了 {len(documents)} 个文档文件")

    # 初始化 ChromaDB（删掉旧集合，避免 embedding 维度不匹配）
    logger.info("初始化 ChromaDB...")
    db = chromadb.PersistentClient(path="./index")
    try:
        db.delete_collection("cleaning_robot_kb")
    except (ValueError, chromadb.errors.NotFoundError):
        pass  # 集合不存在，忽略
    collection = db.create_collection("cleaning_robot_kb")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 构建索引
    logger.info("构建向量索引（耗时取决于文档量）...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    logger.info("索引构建完成！")


if __name__ == "__main__":
    build_index()
