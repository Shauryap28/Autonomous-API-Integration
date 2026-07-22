"""
chunker — split fetched doc text into section-aware LangChain Documents.

  1. Split at every markdown heading (## / ### / ####) into sections.
  2. Tag each chunk with its enclosing ENDPOINT (nearest h2 '##'), so retrieval can be
     scoped to one endpoint and ignore the others on the same page.
  3. A section longer than CHUNK_MAX_CHARS is sub-split with LangChain's
     RecursiveCharacterTextSplitter.
  4. No headings at all -> pure recursive splitting fallback.

Metadata per chunk: {doc_url, doc_name, endpoint_section, section_title, chunk_index}.
`doc_url` is the precise identifier (doc_name is only the URL's last segment and is NOT
unique across URLs) — it is what lets us delete or refresh one document's chunks.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

_PAGE = "(page)"


def _heading_level(line):
    s = line.lstrip()
    n = 0
    while n < len(s) and s[n] == "#":
        n += 1
    if n and n < len(s) and s[n] == " ":
        return n
    return 0


def _split_into_sections(text):
    sections = []
    title, endpoint, buf = "(intro)", _PAGE, []
    current_h2 = _PAGE
    for line in text.splitlines():
        level = _heading_level(line)
        if level:
            if buf:
                sections.append((title, endpoint, "\n".join(buf).strip()))
                buf = []
            title = line.lstrip()[level:].strip()
            if level <= 1:
                current_h2 = _PAGE
            elif level == 2:
                current_h2 = title
            endpoint = current_h2
        else:
            buf.append(line)
    if buf:
        sections.append((title, endpoint, "\n".join(buf).strip()))
    return [(t, e, b) for t, e, b in sections if b]


def chunk(text, doc_name, doc_url=""):
    sections = _split_into_sections(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    docs = []

    def _add(content, section_title, endpoint_section):
        docs.append(Document(
            page_content=content,
            metadata={
                "doc_url": doc_url or doc_name,
                "doc_name": doc_name,
                "endpoint_section": endpoint_section,
                "section_title": section_title,
            },
        ))

    if not sections:
        for piece in splitter.split_text(text):
            _add(piece, "(unstructured)", _PAGE)
    else:
        for section_title, endpoint_section, body in sections:
            if len(body) <= settings.CHUNK_MAX_CHARS:
                _add(body, section_title, endpoint_section)
            else:
                for piece in splitter.split_text(body):
                    _add(piece, section_title, endpoint_section)

    for i, d in enumerate(docs):
        d.metadata["chunk_index"] = i
    return docs