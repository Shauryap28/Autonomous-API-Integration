"""Persistent ChromaDB vector store (HNSW cosine), plus store management."""
from collections import defaultdict

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
    """Remove ALL chunks from the store."""
    data = vectorstore.get()
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


# ---------------------------------------------------------------- management

def list_docs(vectorstore):
    """What documents are currently indexed? -> [{doc_url, doc_name, chunks}]"""
    data = vectorstore.get(include=["metadatas"])
    grouped = defaultdict(lambda: {"doc_name": "", "chunks": 0})
    for meta in data.get("metadatas", []) or []:
        url = (meta or {}).get("doc_url") or (meta or {}).get("doc_name") or "(unknown)"
        grouped[url]["doc_name"] = (meta or {}).get("doc_name", "")
        grouped[url]["chunks"] += 1
    return [
        {"doc_url": url, "doc_name": info["doc_name"], "chunks": info["chunks"]}
        for url, info in sorted(grouped.items(), key=lambda kv: -kv[1]["chunks"])
    ]


def delete_doc(vectorstore, doc_url):
    """Remove every chunk belonging to ONE document. Returns how many were removed."""
    data = vectorstore.get(where={"doc_url": doc_url})
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def has_doc(vectorstore, doc_url):
    """Is this document already indexed? (groundwork for doc caching in 4.3)"""
    data = vectorstore.get(where={"doc_url": doc_url}, limit=1)
    return bool(data.get("ids"))


# ---------------------------------------------------------------- retrieval

def get_section_chunks(vectorstore, section_names):
    """All chunk texts for one OR MORE sections, in document order (metadata filter)."""
    if isinstance(section_names, str):
        section_names = [section_names]
    if not section_names:
        return []

    where = ({"endpoint_section": section_names[0]} if len(section_names) == 1
             else {"endpoint_section": {"$in": section_names}})

    data = vectorstore.get(where=where, include=["documents", "metadatas"])
    rows = list(zip(data.get("documents", []), data.get("metadatas", [])))
    rows.sort(key=lambda r: (r[1] or {}).get("chunk_index", 0))
    return [doc for doc, _ in rows]


def get_endpoint_chunks(vectorstore, endpoint_section):
    return get_section_chunks(vectorstore, endpoint_section)