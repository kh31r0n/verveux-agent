import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..db.postgres import get_pool
from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a security knowledge retrieval assistant.
You have been given relevant context documents retrieved from the internal knowledge base.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain enough information to answer, say so clearly.
Respond in the same language the user writes in.
"""

_TOP_K = 5


async def rag_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rag")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rag_node",
        metadata={"thread_id": thread_id, "node": "rag"},
    )

    # Extract the latest user question
    user_question = ""
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None)
        if role == "human" or (hasattr(msg, "role") and msg.role == "user"):
            user_question = msg.content if hasattr(msg, "content") else str(msg)
            break
    if not user_question:
        user_question = state["messages"][-1].content if state["messages"] else ""

    # Generate embedding
    embedding_gen = trace.generation(
        name="rag_embedding",
        model="text-embedding-3-small",
        input={"text": user_question},
    )
    embedding_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question,
    )
    embedding_vector: list[float] = embedding_response.data[0].embedding
    embedding_gen.end(
        usage={"input": embedding_response.usage.prompt_tokens, "output": 0},
    )

    # Query pgvector for top-K similar documents
    pool = await get_pool()
    vector_literal = "[" + ",".join(str(v) for v in embedding_vector) + "]"
    rows = await pool.fetch(
        """
        SELECT content, metadata,
               1 - (embedding <=> $1::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vector_literal,
        _TOP_K,
    )

    context_docs: list[str] = [row["content"] for row in rows]
    logger.info("rag_retrieved", thread_id=thread_id, count=len(context_docs))

    # Build augmented prompt
    context_block = "\n\n---\n\n".join(context_docs) if context_docs else "No relevant documents found."
    messages_payload = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from knowledge base:\n\n{context_block}\n\n"
                f"User question: {user_question}"
            ),
        },
    ]

    generation = trace.generation(
        name="rag_llm",
        model="gpt-4o",
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = await client.chat.completions.create(
        model="gpt-5",
        messages=messages_payload,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "context": context_docs,
    }
