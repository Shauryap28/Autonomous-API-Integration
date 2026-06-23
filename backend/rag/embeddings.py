"""Local BGE-small embeddings (free, no API key). Reused from MDIS."""
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )