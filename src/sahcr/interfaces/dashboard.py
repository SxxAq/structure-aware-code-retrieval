"""Streamlit Interactive Web Dashboard for Structure-Aware Hybrid Code Retrieval (SAHCR)."""

import streamlit as st
import time
from pathlib import Path
import pandas as pd

from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer
from sahcr.indexing.index_pipeline import IndexPipeline
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.ast_reranker import ASTRerankedRetriever
from sahcr.reranking.structural_features import StructuralFeatureExtractor

st.set_page_config(
    page_title="SAHCR - Local Code Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Header
st.title("🔎 Structure-Aware Hybrid Code Retrieval (SAHCR)")
st.caption(
    "MSc Artificial Intelligence & Machine Learning • Minor Project | "
    "**Student:** MD Saalim Aqueel | **Supervisor:** Dr. Rafat Parveen"
)

# Load pipeline with caching
@st.cache_resource(show_spinner="Loading indices & CodeBERT model...")
def load_system(indices_dir: str = "data/indices", db_path: str = "data/indices/metadata.sqlite"):
    store = MetadataStore(db_path)
    pipeline = IndexPipeline(indices_dir=indices_dir)
    loaded = pipeline.load_existing()
    return store, pipeline, loaded


indices_dir = "data/indices"
db_path = "data/indices/metadata.sqlite"
store, pipeline, indices_loaded = load_system(indices_dir, db_path)

# Sidebar
st.sidebar.header("⚙️ Configuration & Index Status")
if indices_loaded:
    total_funcs = store.count()
    py_funcs = store.count("python")
    js_funcs = store.count("javascript")
    st.sidebar.success("✅ Indices Loaded")
    st.sidebar.metric("Total Indexed Functions", total_funcs)
    st.sidebar.write(f"- 🐍 Python: `{py_funcs}`")
    st.sidebar.write(f"- 🟨 JavaScript: `{js_funcs}`")
else:
    st.sidebar.warning("⚠️ No serialized indices found in `data/indices/`.")
    st.sidebar.info("Run `sahcr index <repo_path>` in your terminal to index a codebase.")

st.sidebar.divider()
st.sidebar.subheader("Hyperparameters")
alpha_param = st.sidebar.slider("Reranker Alpha (RRF Weight)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
rrf_k_param = st.sidebar.slider("RRF Smoothing (k)", min_value=10, max_value=100, value=60, step=5)

# Tabs
tab_search, tab_ast, tab_about = st.tabs(["🔍 Code Search", "🌳 AST Inspector", "📖 About & Architecture"])

with tab_search:
    st.subheader("Natural Language Code Search")
    query = st.text_input(
        "Enter natural language intent or task description:",
        value="parse authorization token from header",
        placeholder="e.g. parse bearer authorization token from header value"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        search_mode = st.radio(
            "Search View:",
            ["Proposed System (Hybrid + AST Reranking)", "Compare All 4 Baselines Side-by-Side"],
            horizontal=True
        )
    with col2:
        top_k = st.slider("Top K results:", min_value=1, max_value=10, value=3)

    if st.button("Search Codebase", type="primary") or query:
        if not indices_loaded:
            st.error("Cannot perform search: No indices loaded.")
        else:
            # Build retrievers
            lexical = LexicalRetriever(pipeline.bm25_indexer, metadata_store=store)
            dense = DenseRetriever(pipeline.encoder, pipeline.faiss_indexer, metadata_store=store)
            hybrid = HybridRetriever(lexical, dense, metadata_store=store, rrf_k=rrf_k_param)
            ast_retriever = ASTRerankedRetriever(hybrid, metadata_store=store, alpha=alpha_param)

            if search_mode == "Proposed System (Hybrid + AST Reranking)":
                t0 = time.perf_counter()
                resp = ast_retriever.retrieve(query, top_k=top_k)
                latency = resp.latency_ms

                st.markdown(f"**Found {len(resp.results)} candidate functions** *(Latency: {latency:.2f} ms)*")

                for res in resp.results:
                    unit = res.unit
                    with st.expander(f"🏆 Rank #{res.rank} | {res.unit_id} (Score: {res.score:.4f})", expanded=(res.rank == 1)):
                        if unit:
                            p_str = ", ".join([f"`{p.name}: {p.type_annotation}`" if p.type_annotation else f"`{p.name}`" for p in unit.parameters]) or "None"
                            st.markdown(f"**Function:** `{unit.name}` | **Language:** `{unit.language.title()}` | **Returns:** `{unit.return_annotation or 'None'}`")
                            st.markdown(f"**Parameters:** {p_str}")
                            if unit.docstring:
                                st.info(f"📝 **Docstring:** {unit.docstring}")
                            if unit.ast_node_types:
                                st.caption(f"AST Nodes: `{'`, `'.join(unit.ast_node_types)}`")
                            st.code(unit.source_code, language=unit.language)
                        else:
                            st.write(f"ID: {res.unit_id}")

            else:
                # Compare all 4
                st.subheader("Side-by-Side Controlled Comparison")
                c1, c2, c3, c4 = st.columns(4)

                configs = [
                    ("Config A: BM25", lexical, c1),
                    ("Config B: CodeBERT", dense, c2),
                    ("Config C: Hybrid RRF", hybrid, c3),
                    ("Config D: Hybrid + AST", ast_retriever, c4),
                ]

                for name, retriever, col in configs:
                    resp = retriever.retrieve(query, top_k=top_k)
                    with col:
                        st.markdown(f"### {name}")
                        st.caption(f"⏱️ {resp.latency_ms:.1f} ms")
                        for r in resp.results:
                            st.markdown(f"**#{r.rank}** `{r.unit.name if r.unit else r.unit_id}`")
                            st.caption(f"Score: `{r.score:.3f}` | File: `{r.unit.file_path if r.unit else ''}`")
                            if r.unit:
                                st.code(r.unit.source_code[:180] + "..." if len(r.unit.source_code) > 180 else r.unit.source_code, language=r.unit.language)
                            st.divider()

with tab_ast:
    st.subheader("Tree-sitter AST Structure Inspector")
    if indices_loaded:
        all_units = list(store.iter_all_units())
        if all_units:
            unit_options = {u.id: u for u in all_units}
            selected_id = st.selectbox("Select a function to inspect:", list(unit_options.keys()))
            u = unit_options[selected_id]

            st.markdown(f"### `{u.name}` ({u.language.title()})")
            st.write(f"- **File:** `{u.file_path}` (Lines {u.start_line}–{u.end_line})")
            st.write(f"- **Async:** `{u.is_async}` | **Class Method:** `{u.is_method}` (Class: `{u.parent_class or 'N/A'}`)")
            st.write(f"- **Return Annotation:** `{u.return_annotation or 'None'}`")

            st.markdown("#### Parameters Extracted via AST")
            if u.parameters:
                p_df = pd.DataFrame([p.model_dump() for p in u.parameters])
                st.dataframe(p_df, use_container_width=True)
            else:
                st.write("No parameters.")

            st.markdown("#### Structural AST Node Types")
            st.write(u.ast_node_types)

            st.markdown("#### Normalized Sub-tokens")
            st.write(f"Function Name Sub-tokens: `{u.normalized_name_tokens}`")
            st.write(f"Parameter Tokens: `{u.parameter_tokens}`")
            st.write(f"Return Tokens: `{u.return_tokens}`")

            st.code(u.source_code, language=u.language)
        else:
            st.info("No units found in store.")
    else:
        st.info("Load or build indices first.")

with tab_about:
    st.subheader("System Architecture")
    st.markdown("""
    **Structure-Aware Hybrid Code Retrieval (SAHCR)** combines:
    1. **Lexical Retrieval (BM25):** Fast exact keyword and identifier matching.
    2. **Dense Semantic Retrieval (CodeBERT + FAISS):** Intent embeddings mapped into a 768-dimensional vector space.
    3. **Reciprocal Rank Fusion (RRF):** Merging lexical and dense candidates into a balanced pool.
    4. **AST Structural Reranking:** Precision re-ranking using Abstract Syntax Trees to eliminate test mocks, comments, and peripheral functions.
    """)
