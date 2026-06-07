from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from config import get_settings
from services.index_service import get_vector_index
from services.llm_service import get_llm
from services.document_service import get_document_count

# System prompt template
SYSTEM_PROMPT = """你是一个专业的企业知识库助手。请基于提供的上下文文档回答用户问题。

规则：
1. 仅基于提供的上下文文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"文档中未找到相关信息"
3. 回答中引用来源时，使用 [来源: 文件名] 格式标注
4. 使用Markdown格式组织回答，合理使用标题、列表、代码块

上下文文档：
{context}"""

# In-memory chat memory (single session)
_chat_memory: ChatMemoryBuffer | None = None


def get_chat_memory() -> ChatMemoryBuffer:
    global _chat_memory
    if _chat_memory is None:
        settings = get_settings()
        _chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=settings.chat_token_limit,
        )
    return _chat_memory


def reset_chat_memory() -> None:
    global _chat_memory
    _chat_memory = None


def rewrite_query(question: str) -> str:
    """Rewrite user query into a more retrieval-friendly form."""
    llm = get_llm()
    rewrite_prompt = PromptTemplate(
        "你是一个查询重写助手。请将用户的口语化提问重写为更适合知识库检索的形式。\n"
        "要求：保持原始意图，使用更正式和完整的表述，不要回答问题。\n"
        "原始问题：{question}\n"
        "重写后的问题："
    )
    response = llm.predict(rewrite_prompt, question=question)
    return response.strip()


def retrieve_basic(query: str) -> list[NodeWithScore]:
    """Basic vector similarity retrieval."""
    settings = get_settings()
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=settings.top_k)
    return retriever.retrieve(query)


def retrieve_hyde(query: str) -> list[NodeWithScore]:
    """HyDE: generate hypothetical document, then retrieve with its embedding."""
    settings = get_settings()
    llm = get_llm()
    index = get_vector_index()

    # Generate hypothetical answer
    hyde_prompt = PromptTemplate(
        "请基于你的知识，简要回答以下问题。如果你不确定，请给出合理的猜测：\n{query}"
    )
    hypothetical_answer = llm.predict(hyde_prompt, query=query)

    # Retrieve using the hypothetical answer as the query
    retriever = index.as_retriever(similarity_top_k=settings.top_k)
    return retriever.retrieve(hypothetical_answer)


def retrieve_window(query: str) -> list[NodeWithScore]:
    """Sentence Window retrieval: retrieve small chunks, expand with surrounding context."""
    settings = get_settings()
    index = get_vector_index()

    # Create query engine with MetadataReplacementPostProcessor
    query_engine = index.as_query_engine(
        similarity_top_k=settings.top_k,
        node_postprocessors=[
            MetadataReplacementPostProcessor(
                target_metadata_key="window",
            ),
        ],
    )
    response = query_engine.retrieve(query)
    return response


def rerank_nodes(
    nodes: list[NodeWithScore], query: str
) -> list[NodeWithScore]:
    """Call Docker-hosted reranker service (TEI / Infinity compatible)."""
    import httpx

    settings = get_settings()
    texts = [node.node.get_content() for node in nodes]

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                settings.reranker_url,
                json={"query": query, "passages": texts},
            )
            resp.raise_for_status()
            payload = resp.json()
            # payload = {"results": [{"index": ..., "passage": ..., "score": ...}, ...]}
            results = payload.get("results", payload)
    except Exception as e:
        print(f"Reranker 调用失败，跳过重排: {e}")
        return nodes

    # Build score lookup and reorder
    score_map = {r["index"]: r["score"] for r in results}
    for i, node in enumerate(nodes):
        node.score = score_map.get(i, node.score)

    nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
    return nodes[:settings.top_n]


def retrieve(
    question: str, strategy: str = "basic", use_rerank: bool = False
) -> list[NodeWithScore]:
    """Full retrieval pipeline: rewrite -> strategy -> optional rerank."""
    # Step 1: Query rewriting
    rewritten = rewrite_query(question)

    # Step 2: Strategy-based retrieval
    if strategy == "hyde":
        nodes = retrieve_hyde(rewritten)
    elif strategy == "window":
        nodes = retrieve_window(rewritten)
    else:
        nodes = retrieve_basic(rewritten)

    # Step 3: Optional reranking
    if use_rerank and nodes:
        nodes = rerank_nodes(nodes, rewritten)

    return nodes


def build_prompt(question: str, nodes: list[NodeWithScore]) -> str:
    """Assemble the full prompt with context, memory, and user question."""
    memory = get_chat_memory()

    # Build context string from retrieved nodes
    context_parts = []
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source_file", "未知来源")
        text = node.node.get_content()
        context_parts.append(f"[文档 {i} | 来源: {source}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # Build chat history string
    history_msgs = memory.get()
    history_text = ""
    if history_msgs:
        parts = []
        for msg in history_msgs:
            role = "用户" if msg.role == "user" else "助手"
            parts.append(f"{role}: {msg.content}")
        history_text = "\n".join(parts)

    # Assemble full prompt
    system = SYSTEM_PROMPT.format(context=context)
    sections = [system]
    if history_text:
        sections.append(f"会话历史：\n{history_text}")
    sections.append(f"用户问题：{question}")
    return "\n\n".join(sections)


def get_sources(nodes: list[NodeWithScore]) -> list[dict]:
    """Extract source info from retrieved nodes for frontend display."""
    sources = []
    for node in nodes:
        sources.append({
            "source_file": node.node.metadata.get("source_file", "未知"),
            "score": round(node.score, 4) if node.score else 0.0,
        })
    return sources


def chat(
    question: str, strategy: str = "basic", use_rerank: bool = False
) -> tuple:
    """
    Execute full RAG pipeline.
    Returns: (prompt: str, sources: list[dict], has_knowledge: bool)
    """
    if get_document_count() == 0:
        return (
            "当前知识库为空，请先上传文档。",
            [],
            False,
        )

    # Retrieve relevant nodes
    settings = get_settings()
    nodes = retrieve(question, strategy, use_rerank)
    print(f"[RAG] strategy={strategy} | rerank={use_rerank} | "
          f"top_k={settings.top_k} | top_n={settings.top_n} | "
          f"retrieved={len(nodes)} nodes | scores=["
          + ", ".join(f"{n.score:.4f}" for n in nodes[:5])
          + ("...]" if len(nodes) > 5 else "]"))

    # Build prompt
    prompt = build_prompt(question, nodes)

    # Extract sources
    sources = get_sources(nodes)

    # Add user message to memory
    memory = get_chat_memory()
    memory.put(ChatMessage(content=question, role=MessageRole.USER))

    return prompt, sources, True
