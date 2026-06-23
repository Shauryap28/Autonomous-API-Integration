"""
retriever — similarity retrieval, optionally scoped to one endpoint.

Divergence from MDIS (documented): MDIS used MMR to diversify Q&A context. Here
we want the MOST relevant chunks for a single aspect (auth / params / pagination),
so plain similarity beats MMR — diversity would risk dropping the one chunk we need.

When endpoint_section is given, ChromaDB applies it as a `where` filter DURING the
search, so we never see the other ~30 endpoints on the page. Same fix as MDIS's
doc_name scoping, one level down (endpoint instead of document).
"""
from backend.config import settings


def get_retriever(vectorstore, endpoint_section=None):
    search_kwargs = {"k": settings.TOP_K}
    if endpoint_section:
        search_kwargs["filter"] = {"endpoint_section": endpoint_section}
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)