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


def get_section_chunks(vectorstore, section_names):
    """Return all chunk texts for one OR MORE endpoint/section names, in doc order.

    A metadata `where` filter (not a similarity search). Accepts a single name or a
    list; for several sections we use Chroma's `$in` operator.
    """
    if isinstance(section_names, str):
        section_names = [section_names]
    if not section_names:
        return []

    if len(section_names) == 1:
        where = {"endpoint_section": section_names[0]}
    else:
        where = {"endpoint_section": {"$in": section_names}}

    data = vectorstore.get(where=where, include=["documents", "metadatas"])
    rows = list(zip(data.get("documents", []), data.get("metadatas", [])))
    rows.sort(key=lambda r: (r[1] or {}).get("chunk_index", 0))
    return [doc for doc, _ in rows]


# Backwards-compatible alias (older callers used get_endpoint_chunks for one section).
def get_endpoint_chunks(vectorstore, endpoint_section):
    return get_section_chunks(vectorstore, endpoint_section)