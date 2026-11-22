"""Interactive Rich + Typer CLI for SAHCR local code search."""

import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

from sahcr.indexing.metadata_store import MetadataStore
from sahcr.indexing.bm25_indexer import BM25Indexer
from sahcr.indexing.dense_indexer import CodeBERTEncoder, FAISSIndexer
from sahcr.indexing.index_pipeline import IndexPipeline
from sahcr.parsers.ingester import RepositoryIngester
from sahcr.retrieval.retrievers import LexicalRetriever, DenseRetriever
from sahcr.retrieval.hybrid_retriever import HybridRetriever
from sahcr.reranking.ast_reranker import ASTRerankedRetriever
from sahcr.evaluation.dataloader import BenchmarkDataLoader
from sahcr.evaluation.benchmark_runner import BenchmarkRunner

app = typer.Typer(
    name="sahcr",
    help="Structure-Aware Hybrid Code Retrieval (SAHCR) CLI",
    add_completion=False
)
console = Console()


def _get_pipeline(indices_dir: str = "data/indices", db_path: str = "data/indices/metadata.sqlite"):
    """Loads metadata store and existing indices."""
    store = MetadataStore(db_path)
    pipeline = IndexPipeline(indices_dir=indices_dir)
    loaded = pipeline.load_existing()
    return store, pipeline, loaded


@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Path to code repository to index"),
    indices_dir: str = typer.Option("data/indices", help="Directory to save indices"),
    db_path: str = typer.Option("data/indices/metadata.sqlite", help="Path to SQLite metadata store"),
    batch_size: int = typer.Option(16, help="Batch size for CodeBERT encoding"),
    pooling: str = typer.Option("cls", help="Pooling strategy: 'cls' or 'mean'")
):
    """Parses code units with Tree-sitter, extracts AST metadata, and builds BM25 and FAISS indices."""
    console.print(f"[bold cyan]🔍 Ingesting repository:[/bold cyan] {repo_path}")
    t0 = time.perf_counter()

    store = MetadataStore(db_path)
    ingester = RepositoryIngester(metadata_store=store)
    units = ingester.ingest_path(repo_path, save_to_store=True)

    if not units:
        console.print("[yellow]⚠️ No supported Python or JavaScript functions found.[/yellow]")
        return

    console.print(f"[green]✓ Extracted {len(units)} functions and AST structural metadata.[/green]")

    console.print("[bold cyan]🔨 Building BM25 and CodeBERT + FAISS indices...[/bold cyan]")
    pipeline = IndexPipeline(indices_dir=indices_dir, pooling=pooling)
    pipeline.build_from_units(units, batch_size=batch_size)

    elapsed = time.perf_counter() - t0
    console.print(f"[bold green]✨ Indexing complete in {elapsed:.2f}s! Indices saved to {indices_dir}[/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    method: str = typer.Option("ast_reranked", help="Retrieval method: 'bm25', 'dense', 'hybrid', 'ast_reranked'"),
    top_k: int = typer.Option(5, help="Number of results to display"),
    indices_dir: str = typer.Option("data/indices", help="Directory with index files"),
    db_path: str = typer.Option("data/indices/metadata.sqlite", help="Path to SQLite metadata store")
):
    """Executes a code search query using the specified retrieval configuration."""
    store, pipeline, loaded = _get_pipeline(indices_dir, db_path)
    if not loaded:
        console.print("[bold red]❌ Indices not found! Run 'sahcr index <repo_path>' first.[/bold red]")
        raise typer.Exit(1)

    lexical = LexicalRetriever(pipeline.bm25_indexer, metadata_store=store)
    dense = DenseRetriever(pipeline.encoder, pipeline.faiss_indexer, metadata_store=store)
    hybrid = HybridRetriever(lexical, dense, metadata_store=store)
    ast_retriever = ASTRerankedRetriever(hybrid, metadata_store=store)

    retrievers = {
        "bm25": lexical,
        "dense": dense,
        "hybrid": hybrid,
        "ast_reranked": ast_retriever,
    }

    selected = retrievers.get(method.lower())
    if not selected:
        console.print(f"[red]Invalid method '{method}'. Choose from: bm25, dense, hybrid, ast_reranked[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Query:[/bold] [italic]{query}[/italic] | [bold]Method:[/bold] [cyan]{method.upper()}[/cyan]")
    resp = selected.retrieve(query, top_k=top_k)

    console.print(f"[dim]Latency: {resp.latency_ms:.2f} ms | Returned: {len(resp.results)} candidates[/dim]\n")

    if not resp.results:
        console.print("[yellow]No matching code found.[/yellow]")
        return

    for res in resp.results:
        unit = res.unit
        title = f"Rank {res.rank} | Score: {res.score:.4f} | {res.unit_id}"
        if unit:
            lang = unit.language
            params = ", ".join([f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name for p in unit.parameters])
            ret = f" -> {unit.return_annotation}" if unit.return_annotation else ""
            sig = f"[bold cyan]Signature:[/bold cyan] {unit.name}({params}){ret}"
            ast_nodes = f"[dim]AST Nodes: {', '.join(unit.ast_node_types[:6])}[/dim]" if unit.ast_node_types else ""
            
            syntax = Syntax(unit.source_code, lang, theme="monokai", line_numbers=True, start_line=unit.start_line)
            content = Text.from_markup(f"{sig}\n{ast_nodes}\n\n")
            console.print(Panel(syntax, title=title, subtitle=sig, border_style="green" if res.rank == 1 else "blue"))
        else:
            console.print(Panel(Text(f"Unit ID: {res.unit_id}"), title=title))


@app.command()
def compare(
    query: str = typer.Argument(..., help="Query to run across all 4 configurations"),
    top_k: int = typer.Option(3, help="Top K results per configuration"),
    indices_dir: str = typer.Option("data/indices", help="Directory with index files"),
    db_path: str = typer.Option("data/indices/metadata.sqlite", help="Path to SQLite metadata store")
):
    """Runs a single query across all 4 configurations to compare ranking changes."""
    store, pipeline, loaded = _get_pipeline(indices_dir, db_path)
    if not loaded:
        console.print("[bold red]❌ Indices not found! Run 'sahcr index <repo_path>' first.[/bold red]")
        raise typer.Exit(1)

    lexical = LexicalRetriever(pipeline.bm25_indexer, metadata_store=store)
    dense = DenseRetriever(pipeline.encoder, pipeline.faiss_indexer, metadata_store=store)
    hybrid = HybridRetriever(lexical, dense, metadata_store=store)
    ast_retriever = ASTRerankedRetriever(hybrid, metadata_store=store)

    configs = [
        ("Config A (BM25)", lexical),
        ("Config B (CodeBERT)", dense),
        ("Config C (Hybrid RRF)", hybrid),
        ("Config D (Hybrid + AST)", ast_retriever),
    ]

    table = Table(title=f"Comparative Retrieval for: '{query}'", show_lines=True)
    table.add_column("Rank", style="bold yellow", width=6)
    table.add_column("Config A: BM25", style="white")
    table.add_column("Config B: CodeBERT", style="cyan")
    table.add_column("Config C: Hybrid RRF", style="magenta")
    table.add_column("Config D: Proposed (AST)", style="bold green")

    results_by_config = {}
    latencies = {}
    for name, ret in configs:
        resp = ret.retrieve(query, top_k=top_k)
        results_by_config[name] = resp.results
        latencies[name] = resp.latency_ms

    for r in range(top_k):
        row = [f"#{r + 1}"]
        for name, _ in configs:
            cand_list = results_by_config[name]
            if r < len(cand_list):
                cand = cand_list[r]
                row.append(f"{cand.unit_id}\n(Score: {cand.score:.3f})")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)
    console.print("\n[bold]Execution Latencies:[/bold]")
    for name, lat in latencies.items():
        console.print(f"  • {name}: {lat:.2f} ms")


@app.command()
def evaluate(
    queries_path: str = typer.Argument(..., help="Path to benchmark queries JSONL file"),
    indices_dir: str = typer.Option("data/indices", help="Directory with index files"),
    db_path: str = typer.Option("data/indices/metadata.sqlite", help="Path to SQLite metadata store"),
    top_k: int = typer.Option(10, help="Cutoff K for metrics")
):
    """Runs full benchmark evaluation across Configurations A, B, C, D and reports statistical significance."""
    store, pipeline, loaded = _get_pipeline(indices_dir, db_path)
    if not loaded:
        console.print("[bold red]❌ Indices not found! Run 'sahcr index <repo_path>' first.[/bold red]")
        raise typer.Exit(1)

    queries = BenchmarkDataLoader.load_jsonl(queries_path)
    console.print(f"[bold cyan]📊 Loaded {len(queries)} evaluation queries.[/bold cyan]")

    lexical = LexicalRetriever(pipeline.bm25_indexer, metadata_store=store)
    dense = DenseRetriever(pipeline.encoder, pipeline.faiss_indexer, metadata_store=store)
    hybrid = HybridRetriever(lexical, dense, metadata_store=store)
    ast_retriever = ASTRerankedRetriever(hybrid, metadata_store=store)

    runner = BenchmarkRunner(lexical, dense, hybrid, ast_retriever)
    eval_results = runner.run_evaluation(queries, top_k=top_k)
    reports = eval_results["reports"]

    table = Table(title="Controlled Ablation Evaluation Results", show_lines=True)
    table.add_column("Configuration", style="bold")
    table.add_column("MRR@10", justify="right")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Latency P50", justify="right")
    table.add_column("Latency P90", justify="right")

    for name, rep in reports.items():
        table.add_row(
            name,
            f"{rep.mrr_at_10:.4f}",
            f"{rep.recall_at_1:.4f}",
            f"{rep.recall_at_5:.4f}",
            f"{rep.recall_at_10:.4f}",
            f"{rep.ndcg_at_10:.4f}",
            f"{rep.latency_p50_ms:.1f}ms",
            f"{rep.latency_p90_ms:.1f}ms"
        )
    console.print(table)

    sig = eval_results["significance_d_vs_c"]
    console.print("\n[bold]Statistical Significance (Config D vs Config C):[/bold]")
    console.print(f"  • Metric: {sig.metric} (Gain: {sig.delta_percentage:+.2f}%)")
    console.print(f"  • Wilcoxon p-value: {sig.wilcoxon_p_value:.5f} (Significant at p<0.05: [bold {'green' if sig.is_significant_05 else 'red'}]{sig.is_significant_05}[/])")


if __name__ == "__main__":
    app()
