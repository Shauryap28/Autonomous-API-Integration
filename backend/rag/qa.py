"""
qa — question answering over the indexed API documentation.

Lets someone explore an API's docs ("how does auth work?") BEFORE committing to a goal,
so goal-drafting is informed rather than guesswork.

Two retrieval scopes, because questions come in two shapes:
  • unscoped  -> semantic search across the whole document. Good for "how does X work?"
  • scoped    -> restricted to ONE section via the endpoint_section metadata filter.
                 Good for "summarize the Pokémon (group) endpoint", where an unscoped
                 search returns 5 chunks from anywhere and misses the section entirely.

The scoped path reuses exactly the metadata-filter pattern that fixed cross-endpoint
contamination in Phase 1 — same idea, applied to Q&A.

Returns the answer AND the retrieved chunks, so the UI can show its sources.
"""
from backend.config import settings
from backend.llm import complete_text

_PROMPT = """You are answering questions about an API, using ONLY the documentation excerpts below.

QUESTION: {question}

DOCUMENTATION EXCERPTS:
{context}

Answer concisely and factually from the excerpts. If the excerpts do not contain the
answer, say so plainly rather than guessing. Where useful, mention the endpoint or
section name the answer comes from.
"""


def _build_filter(doc_url, section):
    """Chroma takes a single condition directly, but needs $and for several."""
    conditions = []
    if doc_url:
        conditions.append({"doc_url": doc_url})
    if section:
        conditions.append({"endpoint_section": section})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def answer_question(vectorstore, question, doc_url=None, section=None, k=None):
    """Return (answer, sources). `section` scopes retrieval to one endpoint section."""
    k = k or settings.QA_TOP_K
    search_kwargs = {"k": k}
    where = _build_filter(doc_url, section)
    if where:
        search_kwargs["filter"] = where

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    hits = retriever.invoke(question)

    if not hits:
        scope = f" within '{section}'" if section else ""
        return f"No relevant documentation was found for that question{scope}.", []

    context = "\n\n".join(
        f"[{h.metadata.get('endpoint_section', '?')} / {h.metadata.get('section_title', '?')}]\n"
        f"{h.page_content}"
        for h in hits
    )
    prompt = _PROMPT.format(question=question, context=context[:14000])
    answer = complete_text(prompt, settings.QA_MAX_OUTPUT_TOKENS)
    return answer, hits