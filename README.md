# Structure-Aware Hybrid Code Retrieval (SAHCR)
> **Design and Implementation of a Local Semantic Retrieval System**  
> MSc Artificial Intelligence & Machine Learning (Semester III) Minor Project  
> **Student:** MD Saalim Aqueel | **Supervisor:** Dr. Rafat Parveen | **Session:** 2025–2027

---

## Overview

SAHCR is a local, two-stage code search system that unifies:
1. **Lexical Retrieval:** BM25 over normalized identifiers, docstrings, and syntax keywords.
2. **Dense Semantic Retrieval:** CodeBERT (`microsoft/codebert-base`) embeddings indexed with FAISS-CPU.
3. **Hybrid Rank Fusion:** Reciprocal Rank Fusion (RRF) with $k=60$.
4. **AST Structural Reranking:** Precision re-ranking using Tree-sitter Abstract Syntax Trees for Python and JavaScript, leveraging function signatures, parameters, return annotations, and syntactic node hierarchies.

---

## Repository Layout

```text
├── configs/
│   └── default_config.yaml            # Indexing, model, and reranking parameters
├── src/
│   └── sahcr/
│       ├── models.py                  # Pydantic data models for code units & parameters
│       ├── parsers/
│       │   ├── base_parser.py         # Abstract base parser
│       │   ├── python_parser.py       # Tree-sitter Python AST extractor
│       │   ├── javascript_parser.py   # Tree-sitter JavaScript AST extractor
│       │   ├── normalizer.py          # CamelCase/snake_case sub-token splitting
│       │   └── ingester.py            # Recursive repository code walker & AST ingester
│       ├── indexing/
│       │   └── metadata_store.py      # SQLite store for code snippets & structural attributes
│       ├── retrieval/                 # (Phase 2 & 3) BM25, CodeBERT FAISS, RRF fusion
│       ├── reranking/                 # (Phase 3) AST structural reranker
│       └── interfaces/                # (Phase 5) Rich CLI & Streamlit dashboard
├── tests/
│   ├── test_normalizer.py             # Sub-token splitting & tokenization tests
│   ├── test_parsers.py                # Tree-sitter extraction accuracy tests
│   ├── test_metadata_store.py         # SQLite metadata roundtrip tests
│   └── test_ingester.py               # Repository walking & ingestion tests
└── pyproject.toml
```

---

## Setup & Testing

### 1. Environment Activation
```bash
source .venv/bin/activate
```

### 2. Run Test Suite
```bash
PYTHONPATH=src pytest tests/
```

### 3. Ingest Code Repository
```python
from sahcr.indexing.metadata_store import MetadataStore
from sahcr.parsers.ingester import RepositoryIngester

store = MetadataStore("data/indices/metadata.sqlite")
ingester = RepositoryIngester(metadata_store=store)
units = ingester.ingest_path("/path/to/source/code")
print(f"Extracted {len(units)} functions.")
```

