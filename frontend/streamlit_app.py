"""
Streamlit UI — explore an API's docs, draft a goal, run the agent, inspect the result.

Run from the repo root:
    streamlit run frontend/streamlit_app.py

DESIGN NOTES
- Streamlit re-runs this whole file on EVERY interaction. So:
    * expensive resources (embedding model, vectorstore) use @st.cache_resource;
    * completed work lives in st.session_state and is never repeated;
    * the app is an explicit STAGE MACHINE — each rerun renders the current stage.
- The UI holds NO logic: every action calls an existing backend/ function.
  main.py stays the primary test harness; this is a second entry point.
- Cost ordering: browsing endpoints is free (filter + slice on already-chunked text).
  Embedding happens lazily on first real need (Q&A or a run) via ensure_indexed().
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings
from backend.rag.fetcher import fetch
from backend.rag.chunker import chunk
from backend.rag.embeddings import get_embeddings
from backend.rag.vectorstore import (
    get_vectorstore, add_documents, count, clear, list_docs, delete_doc, has_doc,
)
from backend.rag.qa import answer_question
from backend.rag.extractor import extract_api_schema
from backend.agent.selector import select_endpoint
from backend.agent.global_sections import identify_global_sections
from backend.agent.graph import build_graph

st.set_page_config(page_title="API Integration Engine", page_icon="🔌", layout="wide")


# ----------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def cached_vectorstore():
    return get_vectorstore(get_embeddings())


# ----------------------------------------------------------------- session state

DEFAULTS = {
    "stage": 0,          # 0 no doc | 1 loaded | 2 candidates | 3 schema ready | 4 run done
    "doc_src": "",
    "doc_name": "",
    "fetch_result": None,
    "docs": None,
    "sections": [],
    "goal": "",
    "selection": None,
    "target_endpoint": "",
    "global_sections": [],
    "schema": None,
    "trace": [],
    "final_state": None,
    "qa_answer": None,
    "qa_sources": [],
    "qa_scope": "All sections",
}


def init_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_session():
    for key in DEFAULTS:
        st.session_state.pop(key, None)
    init_state()


init_state()
vs = cached_vectorstore()


# ----------------------------------------------------------------- helpers

def doc_name_from(src):
    return src.rstrip("/").split("/")[-1] or src


def section_names(docs):
    counts = {}
    for d in docs:
        name = d.metadata["endpoint_section"]
        counts[name] = counts.get(name, 0) + 1
    return [(name, n) for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def ensure_indexed():
    """Embed this doc only if it isn't in the store already (idempotent).

    Browsing costs nothing; you pay the embedding cost the first time you actually
    need semantics (Q&A or a run). has_doc() makes repeat calls free — a preview of
    the doc-caching phase.
    """
    src = st.session_state.doc_src
    if has_doc(vs, src):
        return False
    with st.spinner(f"Indexing {len(st.session_state.docs)} chunks (first time for this doc)..."):
        add_documents(vs, st.session_state.docs)
    return True


def merge_update(state, update):
    """Apply a LangGraph node's partial update to our local copy of the state.

    Mirrors LangGraph's merge rules: plain fields overwrite; error_history APPENDS
    (it declares an operator.add reducer in AgentState).
    """
    for key, value in update.items():
        if key == "error_history":
            state[key] = state.get(key, []) + list(value)
        else:
            state[key] = value
    return state


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Documentation")
    src_input = st.text_input(
        "Docs URL or file path",
        value=st.session_state.doc_src or "https://docs.github.com/en/rest/repos/repos",
    )

    if st.button("Load docs", type="primary", use_container_width=True):
        with st.spinner("Fetching and chunking..."):
            fr = fetch(src_input)
        st.session_state.fetch_result = fr
        if fr.looks_thin:
            st.session_state.stage = 0
            st.session_state.docs = None
        else:
            name = doc_name_from(src_input)
            docs = chunk(fr.text, name, doc_url=src_input)
            st.session_state.update(
                doc_src=src_input, doc_name=name, docs=docs,
                sections=section_names(docs), stage=1,
                selection=None, target_endpoint="", schema=None,
                trace=[], final_state=None, qa_answer=None,
            )
        st.rerun()

    if st.button("Reset session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.header("Vector store")
    st.caption(f"{count(vs)} chunks indexed")

    indexed = list_docs(vs)
    if not indexed:
        st.caption("_Empty — no documents indexed yet._")
    else:
        for doc in indexed:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.write(f"**{doc['doc_name'] or '(doc)'}** — {doc['chunks']} chunks")
                st.caption(doc["doc_url"][:55])
            with col_b:
                if st.button("🗑", key=f"del_{doc['doc_url']}", help="Delete this document"):
                    removed = delete_doc(vs, doc["doc_url"])
                    st.toast(f"Removed {removed} chunks")
                    st.rerun()
        if st.button("Clear entire store", use_container_width=True):
            st.toast(f"Cleared {clear(vs)} chunks")
            st.rerun()

    st.divider()
    st.header("Settings")
    # NOTE: these mutate settings in memory for this session only — they do not write
    # back to settings.py. Fine for a local single-user tool; logged as a known wart.
    settings.USE_SANDBOX = st.toggle("Docker sandbox", value=settings.USE_SANDBOX)
    settings.FORCE_FAILURE = st.toggle(
        "Force failure (demo)", value=settings.FORCE_FAILURE,
        help="Deliberately corrupt the first script so the API returns a real 404 — "
             "shows the self-healing loop on demand.",
    )
    settings.MAX_RETRIES = st.slider("Max retries", 1, 8, settings.MAX_RETRIES)
    # Read-only ON PURPOSE: changing the key strategy changes how each record's identity
    # is derived, so previously-stored rows no longer collide -> duplicates instead of
    # updates. Change it in settings.py deliberately, and TRUNCATE the table afterwards.
    st.caption(f"record key strategy: `{settings.KEY_STRATEGY}`  \n"
               f"_(set in settings.py — changing it invalidates idempotency "
               f"against already-stored rows)_")


# ----------------------------------------------------------------- main

st.title("Autonomous API Integration Engine")
fr = st.session_state.fetch_result

# ---------- stage 0: nothing loaded ----------
if st.session_state.stage == 0:
    if fr is not None and fr.looks_thin:
        st.error(
            f"That page returned only {fr.char_count} characters — likely a "
            "JavaScript-rendered shell. Save it as a PDF and pass the file path, or use "
            "an OpenAPI/Swagger or raw-markdown version of the docs."
        )
    else:
        st.info("Load an API documentation page from the sidebar to begin.")
    st.stop()

# ---------- header metrics ----------
c1, c2, c3 = st.columns(3)
c1.metric("Characters", f"{fr.char_count:,}")
c2.metric("Chunks", len(st.session_state.docs))
c3.metric("Sections", len(st.session_state.sections))

tab_explore, tab_run = st.tabs(["Explore the docs", "Run a goal"])

# ================================================================ EXPLORE
with tab_explore:
    st.subheader("Endpoints in this documentation")
    st.caption("Straight from the doc's headings — no LLM, no embedding, instant.")
    for name, n in st.session_state.sections:
        with st.expander(f"{name}  ·  {n} chunks"):
            preview = [d for d in st.session_state.docs
                       if d.metadata["endpoint_section"] == name][:2]
            for d in preview:
                st.caption(d.metadata.get("section_title", ""))
                st.text(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

    st.divider()
    st.subheader("Ask the documentation")
    st.caption("Understand the API before writing a goal. Uses retrieval + the LLM.")

    q_col, s_col = st.columns([3, 2])
    with q_col:
        question = st.text_input(
            "Question", placeholder="How does authentication work?"
        )
    with s_col:
        # Scoping retrieval to ONE section is what makes "summarize section X" work:
        # unscoped search returns k chunks from anywhere and can miss the section.
        scope = st.selectbox(
            "Search within",
            ["All sections"] + [n for n, _ in st.session_state.sections],
            help="Pick a section for questions ABOUT that section "
                 "(e.g. 'summarize this endpoint').",
        )

    if st.button("Ask") and question.strip():
        ensure_indexed()
        section = None if scope == "All sections" else scope
        with st.spinner(f"Searching {'the docs' if section is None else scope}..."):
            answer, sources = answer_question(
                vs, question, doc_url=st.session_state.doc_src, section=section
            )
        st.session_state.qa_answer = answer
        st.session_state.qa_sources = sources
        st.session_state.qa_scope = scope

    if st.session_state.qa_answer:
        if st.session_state.get("qa_scope", "All sections") != "All sections":
            st.caption(f"Scoped to: **{st.session_state.qa_scope}**")
        st.markdown(st.session_state.qa_answer)
        with st.expander(f"Sources ({len(st.session_state.qa_sources)} chunks retrieved)"):
            for h in st.session_state.qa_sources:
                st.caption(
                    f"**{h.metadata.get('endpoint_section', '?')}** / "
                    f"{h.metadata.get('section_title', '?')}"
                )
                st.text(h.page_content[:300] + "...")

# ================================================================ RUN
with tab_run:
    goal = st.text_area(
        "What do you want to fetch?",
        value=st.session_state.goal,
        placeholder="List the first 50 public repositories of the 'github' organization.",
        height=80,
    )

    if st.button("Find matching endpoint", type="primary") and goal.strip():
        st.session_state.goal = goal
        with st.spinner("Matching your goal to a documentation section..."):
            st.session_state.selection = select_endpoint(
                goal, [n for n, _ in st.session_state.sections]
            )
        st.session_state.update(stage=2, target_endpoint="", schema=None,
                                trace=[], final_state=None)
        st.rerun()

    sel = st.session_state.selection

    # ---------- stage 2: candidates or denial ----------
    if sel is not None and not st.session_state.target_endpoint:
        if not sel.can_fulfill or not sel.candidates:
            st.error(f"**No endpoint in these docs can satisfy that goal.**\n\n{sel.reason}")
            st.caption("Available sections:")
            for name, _ in st.session_state.sections[:12]:
                st.caption(f"• {name}")
        else:
            st.success("Candidate endpoints — pick one to continue:")
            for i, c in enumerate(sel.candidates):
                with st.container(border=True):
                    col_a, col_b = st.columns([5, 1])
                    col_a.markdown(f"**{c.section}**  \n`{c.confidence}` — {c.why}")
                    if col_b.button("Use", key=f"cand_{i}"):
                        st.session_state.target_endpoint = c.section
                        ensure_indexed()
                        with st.spinner("Checking for global sections (pagination/auth)..."):
                            st.session_state.global_sections = identify_global_sections(
                                [n for n, _ in st.session_state.sections], exclude=c.section
                            )
                        with st.spinner("Extracting the API schema..."):
                            st.session_state.schema = extract_api_schema(
                                vs, st.session_state.goal, c.section,
                                st.session_state.global_sections,
                            )
                        st.session_state.stage = 3
                        st.rerun()

    # ---------- stage 3+: schema + run ----------
    schema = st.session_state.schema
    if schema is not None:
        st.divider()
        st.subheader("What the agent understood")
        st.caption(f"Target: **{st.session_state.target_endpoint}**")
        if st.session_state.global_sections:
            st.caption(f"Also read: {', '.join(st.session_state.global_sections)}")

        s1, s2 = st.columns(2)
        with s1:
            st.write(f"**Method** `{schema.http_method}`")
            st.write(f"**URL** `{schema.base_url}{schema.endpoint}`")
            st.write(f"**Auth** `{schema.auth_method.value}`")
        with s2:
            st.write(f"**Pagination** `{schema.pagination.type.value}` "
                     f"{schema.pagination.param_names}")
            st.write(f"**Records at** `{schema.response_data_path or '(top-level array)'}`")
            st.write(f"**Params** {len(schema.parameters)}")
        with st.expander("Full schema (JSON)"):
            st.json(json.loads(schema.model_dump_json()))

        if st.button("Run the agent", type="primary"):
            initial_state = {
                "goal": st.session_state.goal,
                "api_schema": schema.model_dump(),
                "current_code": "", "execution_result": {},
                "attempt_number": 0, "max_retries": settings.MAX_RETRIES,
                "error_history": [], "status": "running", "fetched_data": None,
                "rows_upserted": None, "rows_for_endpoint": None, "persist_error": None,
            }
            graph = build_graph(vs, st.session_state.target_endpoint, st.session_state.doc_name)

            trace, state = [], dict(initial_state)
            box = st.container()
            with st.spinner("Running — generate → execute → (diagnose → retry) → persist"):
                # stream_mode="updates" yields {node_name: partial_update} as each node
                # finishes, so we can render the story instead of scraping stdout.
                for step in graph.stream(initial_state, {"recursion_limit": 50},
                                         stream_mode="updates"):
                    for node, update in step.items():
                        state = merge_update(state, update)
                        if node == "generate_code":
                            line = f"**generate_code** → wrote a script (attempt {state['attempt_number']})"
                        elif node == "execute":
                            ex = state.get("execution_result", {})
                            n = len(state["fetched_data"]) if isinstance(state.get("fetched_data"), list) else "-"
                            line = (f"**execute** → exit `{ex.get('exit_code')}`, records `{n}`")
                        elif node == "diagnose_and_fix":
                            last = state["error_history"][-1]["error"] if state.get("error_history") else ""
                            line = f"**diagnose_and_fix** → read the error and rewrote the script  \n`{str(last).strip().splitlines()[-1][:120] if last else ''}`"
                        elif node == "persist_and_verify":
                            line = (f"**persist_and_verify** → upserted `{state.get('rows_upserted')}`; "
                                    f"table holds `{state.get('rows_for_endpoint')}` rows")
                        else:
                            line = f"**{node}**"
                        trace.append(line)
                        box.markdown(f"{len(trace)}. {line}")

            st.session_state.trace = trace
            st.session_state.final_state = state
            st.session_state.stage = 4
            st.rerun()

    # ---------- stage 4: results ----------
    final = st.session_state.final_state
    if final is not None:
        st.divider()
        status = final.get("status", "?")
        st.subheader(f"Outcome: {status.upper()} in {final.get('attempt_number')} attempt(s)")

        t_trace, t_code, t_errors, t_data = st.tabs(
            ["Trace", "Generated code", "Error history", "Data"]
        )

        with t_trace:
            for i, line in enumerate(st.session_state.trace, 1):
                st.markdown(f"{i}. {line}")

        with t_code:
            st.code(final.get("current_code", ""), language="python")
            st.download_button("Download script", final.get("current_code", ""),
                               file_name="fetch_script.py")

        with t_errors:
            history = final.get("error_history", [])
            if not history:
                st.success("No failures — the first attempt worked.")
            for h in history:
                with st.container(border=True):
                    st.write(f"**Attempt {h['attempt']}** — exit `{h['exit_code']}`")
                    st.code(str(h["error"])[:800])

        with t_data:
            if final.get("rows_upserted") is not None:
                d1, d2 = st.columns(2)
                d1.metric("Upserted this run", final["rows_upserted"])
                d2.metric("Rows for this endpoint", final["rows_for_endpoint"],
                          help="Re-run the same goal: this should NOT grow.")
            elif final.get("persist_error"):
                st.error(f"Persistence failed: {final['persist_error']}")

            data = final.get("fetched_data")
            if isinstance(data, list) and data:
                st.caption(f"{len(data)} records fetched")
                st.dataframe(data, use_container_width=True)