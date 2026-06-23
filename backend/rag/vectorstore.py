"""Persistent ChromaDB vector store (HNSW cosine). Reused from MDIS."""
from langchain_chroma import Chroma

from backend.config import settings


def get_vectorstore(embeddings):
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.PERSIST_DIR,
        collection_metadata={"hnsw:space": settings.DISTANCE_METRIC},
    )


def add_documents(vectorstore, documents):
    vectorstore.add_documents(documents)


def count(vectorstore):
    return vectorstore._collection.count()


def clear(vectorstore):
    """Remove ALL chunks (fresh ingest each run for a deterministic measurement)."""
    data = vectorstore.get()
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)


def get_endpoint_chunks(vectorstore, endpoint_section):
    """Return all chunk texts for ONE endpoint, in document order.

    A metadata `where` filter (not a similarity search) — for extraction we want
    the endpoint's COMPLETE documentation, and the section is small (~13 chunks).
    """
    data = vectorstore.get(
        where={"endpoint_section": endpoint_section},
        include=["documents", "metadatas"],
    )
    rows = list(zip(data.get("documents", []), data.get("metadatas", [])))
    rows.sort(key=lambda r: (r[1] or {}).get("chunk_index", 0))
    return [doc for doc, _ in rows]