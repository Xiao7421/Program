"""扫地机器人知识问答 RAG 查询引擎 (ChromaDB + BM25 混合检索 + 中文分词)"""
import json
import os
import re
import logging
import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "你是一个专业的扫地机器人知识问答助手。你的知识来源是扫地机器人的使用指南、"
    "故障排查手册、维护保养说明和安全注意事项。\n\n"
    "回答规则：\n"
    "1. 只回答与扫地机器人相关的问题（使用、故障、维护、选购、安全等）\n"
    "2. 如果问题与扫地机器人无关，请礼貌地说明你仅回答扫地机器人相关问题\n"
    "3. 基于提供的知识库内容回答，不要编造信息\n"
    "4. 如果知识库中没有相关信息，请明确告知用户你不知道\n"
    "5. 回答要详细且实用，给出具体步骤和参数\n"
    "6. 使用友好的语气，用中文回答\n"
    "7. 如果问题涉及安全风险，请务必提醒用户注意安全"
)


TOP_K = 15


def _chinese_tokenizer(text: str) -> list[str]:
    """中文友好的 BM25 分词器：字符二元组 + 英文/数字保留"""
    tokens = []
    for part in re.split(r"(\s+)", text):
        for token in re.findall(r"[a-zA-Z0-9]+|[^\s\w]", part):
            if re.match(r"[一-鿿]", token):
                for i in range(len(token) - 1):
                    tokens.append(token[i : i + 2])
            else:
                tokens.append(token.lower())
    return tokens


# ── 兼容的 BM25 检索器 ──────────────────────────────────


class Bm25Retriever(BaseRetriever):
    """基于 rank_bm25 的自定义检索器，支持中文分词"""

    def __init__(self, nodes, tokenizer, similarity_top_k, **kwargs):
        super().__init__(**kwargs)
        self.similarity_top_k = similarity_top_k
        self._nodes = nodes
        tokenized_corpus = [tokenizer(n.get_content()) for n in nodes]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._tokenizer = tokenizer

    def _retrieve(self, query_bundle):
        query = query_bundle.query_str
        tokenized_query = self._tokenizer(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.similarity_top_k]
        return [
            NodeWithScore(node=self._nodes[i], score=float(scores[i]))
            for i in top_indices
        ]


# ── 问答引擎 ────────────────────────────────────────────


class CleaningRobotQA:
    """扫地机器人 RAG 问答引擎 (ChromaDB + BM25 混合检索)"""

    def __init__(self, index_dir: str = "./index"):
        load_dotenv()
        self.index_dir = index_dir
        self._db = None
        self._collection = None
        self.index = None
        self._nodes = None
        self._bm25_retriever = None
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

        self._init_settings()
        if os.path.exists(index_dir):
            self._load_index()
            self._warm_up_llm()

    def _warm_up_llm(self):
        """首次加载时预热 LLM，避免后续查询因模型加载而超时"""
        try:
            logger.info("预热 LLM（首次加载可能需要较长时间）...")
            resp = Settings.llm.complete("ping")
            logger.info(f"LLM 预热完成（{len(str(resp))} 字符）")
        except Exception as e:
            logger.warning(f"LLM 预热失败（不影响后续使用）: {e}")

    def _init_settings(self):
        """初始化全局设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        Settings.embed_model = DashScopeEmbedding(
            model_name="text-embedding-v4",
            api_key=api_key,
            embed_batch_size=10,
        )
        Settings.llm = Ollama(
            model="qwen2.5:7b",
            request_timeout=300.0,
            temperature=0.3,
            additional_kwargs={"keep_alive": "10m"},
        )

    def _load_index(self):
        """从 ChromaDB 加载索引，并构建 BM25 关键词检索器"""
        logger.info("加载 ChromaDB 索引...")
        self._db = chromadb.PersistentClient(path=self.index_dir)
        self._collection = self._db.get_or_create_collection("cleaning_robot_kb")
        vector_store = ChromaVectorStore(chroma_collection=self._collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        # 从 ChromaDB documents + metadata 构建 BM25 索引
        # (docstore.docs 持久化时为空，需直接从 ChromaDB 提取)
        try:
            collection_data = self._collection.get()
            raw_documents = collection_data["documents"]
            raw_metadatas = collection_data["metadatas"]
            if raw_documents:
                self._nodes = []
                for idx, doc_text in enumerate(raw_documents):
                    meta = raw_metadatas[idx] if raw_metadatas else {}
                    # 从 _node_content 提取 metadata，从 documents 提取文本
                    node_meta = {}
                    nc_str = meta.get("_node_content", "")
                    if nc_str:
                        try:
                            node_dict = json.loads(nc_str)
                            node_meta = node_dict.get("metadata", {})
                        except Exception:
                            node_meta = {k: v for k, v in meta.items() if k != "_node_content"}
                    if not node_meta:
                        node_meta = meta
                    self._nodes.append(TextNode(text=doc_text, metadata=node_meta))

                self._bm25_retriever = Bm25Retriever(
                    nodes=self._nodes,
                    tokenizer=_chinese_tokenizer,
                    similarity_top_k=TOP_K,
                )
                logger.info(
                    f"BM25 索引构建完成（{len(self._nodes)} 个节点，中文二元分词）"
                )
            else:
                logger.warning("ChromaDB 中无文档数据，跳过 BM25")
                self._bm25_retriever = None
        except Exception as e:
            logger.warning(f"BM25 检索器初始化失败，仅使用向量检索: {e}")
            self._bm25_retriever = None

        logger.info("索引加载完成")

    @property
    def is_ready(self) -> bool:
        """索引是否已加载"""
        return self.index is not None

    def reload(self):
        """重新加载索引（重建后调用）"""
        self.memory.reset()
        if os.path.exists(self.index_dir):
            self._load_index()
        else:
            self.index = None
            self._nodes = None
            self._bm25_retriever = None

    def _build_retriever(self):
        """构建混合检索器（向量 + BM25 融合）"""
        vector_retriever = self.index.as_retriever(similarity_top_k=TOP_K)
        if self._bm25_retriever is not None:
            return QueryFusionRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                similarity_top_k=TOP_K,
                num_queries=1,
                mode="reciprocal_rerank",
            )
        return vector_retriever

    def query(self, question: str) -> tuple[str, str]:
        """执行 RAG 查询（向量 + BM25 混合检索），返回 (回答, 来源信息)"""
        retriever = self._build_retriever()

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=self.memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=False,
        )

        try:
            response = chat_engine.chat(question)
        except Exception as e:
            err_str = str(e).lower()
            if "timeout" in err_str or "readtim" in err_str:
                logger.warning("查询超时，重试一次...")
                response = chat_engine.chat(question)
            else:
                raise

        # 提取来源信息
        source_nodes = response.source_nodes
        sources_text = ""
        if source_nodes:
            seen_files = set()
            for node in source_nodes:
                file_name = node.metadata.get("file_name", "未知文件")
                if file_name not in seen_files:
                    seen_files.add(file_name)
                    sources_text += f"\n- {file_name}"
            sources_text = f"参考来源:{sources_text}"

        return str(response), sources_text

    def close(self):
        """释放 ChromaDB 连接，删除目录时避免文件锁冲突"""
        self.index = None
        self._collection = None
        self._db = None
        self._nodes = None
        self._bm25_retriever = None

    def clear_memory(self):
        """清除对话历史"""
        self.memory.reset()
