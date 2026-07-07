import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..db.postgres import get_pool
from ..graphs.state import AgentState
from ..llm import resolve_api_key
from ..providers.openai import OpenAIProvider
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction, latest_user_text

logger = structlog.get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a security knowledge retrieval assistant.
You have been given relevant context documents retrieved from the internal knowledge base.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain enough information to answer, say so clearly.
{language_rule}
"""

_TOP_K = 5


async def rag_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rag")

    # RAG always uses OpenAI for embeddings, for now.
    # The main provider is used for the chat completion.
    openai_provider = OpenAIProvider(config)
    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rag_node",
        metadata={"thread_id": thread_id, "node": "rag"},
    )

    # Extract the user's current turn — all trailing human messages joined,
    # since users often split one question across rapid WhatsApp messages.
    user_question = latest_user_text(state)
    if not user_question:
        user_question = state["messages"][-1].content if state["messages"] else ""

    # Generate embedding
    embedding_gen = trace.generation(
        name="rag_embedding",
        model="text-embedding-3-small",
        input={"text": user_question},
    )
    embedding_response = await openai_provider.embed(
        texts=[user_question],
        model="text-embedding-3-small",
    )
    embedding_vector: list[float] = embedding_response[0]
    embedding_gen.end()

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
    lang_rule = language_instruction(state.get("language", "en"))
    messages_payload = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(language_rule=lang_rule)},
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
        model=model,
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""
    async for chunk in stream:
        write({"type": "token", "content": chunk})
        full_response += chunk

    generation.end(
        output=full_response,
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "context": context_docs,
    }
