from langchain_openai import OpenAIEmbeddings

from app.config import settings


def create_qwen_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.QWEN_EMBEDDING_BASE_URL,
        model=settings.QWEN_EMBEDDING_MODEL,
        chunk_size=20,
        timeout=30.0,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
