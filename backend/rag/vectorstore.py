"""Persistent ChromaDB vector store (HNSW cosine), plus store management.

The store is a PERSISTENT LIBRARY, not a scratchpad: many documents coexist, and
retrieval is always metadata-scoped so they never interfere.
"""
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
    data = vectorstore.get()
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


# ---------------------------------------------------------------- management

def list_docs(vectorstore):
    """Indexed documents -> [{doc_url, doc_name, chunks, embedded_at, content_hash}]"""
    data = vectorstore.get(include=["metadatas"])
    grouped = defaultdict(lambda: {"doc_name": "", "chunks": 0,
                                   "embedded_at": "", "content_hash": ""})
    for meta in data.get("metadatas", []) or []:
        meta = meta or {}
        url = meta.get("doc_url") or meta.get("doc_name") or "(unknown)"
        entry = grouped[url]
        entry["doc_name"] = meta.get("doc_name", "")
        entry["embedded_at"] = meta.get("embedded_at", "")
        entry["content_hash"] = meta.get("content_hash", "")
        entry["chunks"] += 1
    return [dict(doc_url=url, **info)
            for url, info in sorted(grouped.items(), key=lambda kv: -kv[1]["chunks"])]


def delete_doc(vectorstore, doc_url):
    """Remove every chunk of ONE document. Returns how many were removed."""
    data = vectorstore.get(where={"doc_url": doc_url})
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def has_doc(vectorstore, doc_url):
    return bool(vectorstore.get(where={"doc_url": doc_url}, limit=1).get("ids"))


def reindex_doc(vectorstore, doc_url, documents):
    """Replace a document's chunks: DELETE then add.

    Delete-then-add, never append: appending would leave the old and new versions in the
    store together and retrieval would mix stale text with current text.
    """
    removed = delete_doc(vectorstore, doc_url)
    add_documents(vectorstore, documents)
    return removed


def get_doc_sections(vectorstore, doc_url):
    """Section names + chunk counts for an ALREADY-INDEXED document.

    Lets a cached document be loaded without re-fetching or re-chunking it.
    """
    data = vectorstore.get(where={"doc_url": doc_url}, include=["metadatas"])
    counts = defaultdict(int)
    for meta in data.get("metadatas", []) or []:
        counts[(meta or {}).get("endpoint_section", "(page)")] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


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