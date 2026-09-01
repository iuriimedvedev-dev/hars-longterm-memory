"""Importable console entry points for the HARS long-term memory UV project.

``index_main`` / ``mcp_main`` / ``eval_battle_main`` are the pre-existing
entry points (``memory-index`` / ``memory-mcp`` / ``memory-eval-battle`` in
``pyproject.toml``) — unchanged, kept working exactly as before (``memory-mcp``
in particular: the MCP server is live and other agents depend on it).

``main`` is the unified ``memory`` console script (``memory <subcommand>``),
covering servers (MCP, HTTP, gRPC), LightRAG knowledge graph recall, corpus
build/query/eval/regress, and incremental reindex. It imports every heavier
dependency lazily, INSIDE each subcommand handler, not at module scope — so
`memory --help` never pays an import cost for subsystems it doesn't use.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "hars_memory").is_dir():
            return parent
    # Installed wheels do not include the repository's pyproject.toml. The
    # package parent is already importable, but returning it keeps the legacy
    # sys.path bootstrap harmless and, unlike parents[3], valid at any depth.
    return module_path.parent.parent


def _ensure_project_root() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def index_main() -> None:
    _ensure_project_root()
    from hars_memory.server.index import main

    main()


def mcp_main() -> None:
    from hars_memory.mcp_server import main as _mcp_server_main

    _mcp_server_main()


def eval_battle_main() -> None:
    _ensure_project_root()
    from hars_memory.eval.battle import main

    main()


def grpc_main() -> None:
    """Start the gRPC server."""
    from hars_memory.grpc.server import main

    main()


# ---------------------------------------------------------------------------
# `memory` unified subcommand CLI (corpus build/query/eval/regress/status)
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    from hars_memory.corpus.build import (
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        CorpusBuildError,
        build_corpus,
    )

    chunk_size = args.chunk_size if args.chunk_size is not None else DEFAULT_CHUNK_SIZE
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    try:
        result = build_corpus(
            args.paths,
            args.index_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=args.force,
        )
    except CorpusBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Build complete: {result.document_count} documents, {result.chunk_count} chunks")
    print(
        f"  added={result.added} changed={result.changed} "
        f"unchanged={result.unchanged} deleted={result.deleted}"
    )
    print(f"  corpus_fingerprint={result.corpus_fingerprint}")
    print(f"  build_seconds={result.build_seconds:.3f}")
    print(f"  index_dir={result.index_dir}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from hars_memory.corpus.query import DEFAULT_TOP_K, CorpusQueryError, search

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    mode = args.mode or "fusion"
    try:
        hits = search(args.index_dir, args.question, top_k=top_k, mode=mode)
    except CorpusQueryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not hits:
        print("No hits.")
        return 0
    for i, hit in enumerate(hits, start=1):
        print(f"{i}. [{hit.score:.4f}] {hit.file_path}  (chunk={hit.chunk_id}, channel={hit.channel})")
        print(f"   {hit.snippet}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from hars_memory.eval.corpus_eval import DEFAULT_MODE, DEFAULT_TOP_K, CorpusEvalError, run_eval

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    mode = args.mode or DEFAULT_MODE
    try:
        report = run_eval(args.index_dir, args.queries, top_k=top_k, mode=mode)
    except CorpusEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote eval report: {args.report}")
    print(json.dumps(report["metrics"], indent=2))
    return 0


def _cmd_regress(args: argparse.Namespace) -> int:
    from hars_memory.eval.regression import RegressionError, compare_reports

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        verdict = compare_reports(
            baseline,
            candidate,
            baseline_ab_bench_config=args.baseline_ab_bench_config,
            candidate_ab_bench_config=args.candidate_ab_bench_config,
        )
    except RegressionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(verdict.summary())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": verdict.passed,
            "per_metric": [dataclasses.asdict(v) for v in verdict.per_metric],
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote regression report: {args.report}")
    return 0 if verdict.passed else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from hars_memory.corpus.build import MANIFEST_FILENAME

    manifest_path = args.index_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        print(f"ERROR: no {MANIFEST_FILENAME} at {args.index_dir}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_info = manifest["build"]
    documents = manifest["documents"]
    chunk_count = sum(int(d["chunk_count"]) for d in documents.values())

    print(f"index_dir:          {args.index_dir}")
    print(f"documents:          {len(documents)}")
    print(f"chunks:             {chunk_count}")
    print(f"corpus_fingerprint: {build_info['corpus_fingerprint']}")
    print(f"chunk_store_sha256: {build_info['chunk_store_sha256']}")
    print(f"created_at:         {build_info['created_at']}")
    print(f"chunk_size/overlap: {build_info['chunk_size']}/{build_info['chunk_overlap']}")
    print(f"tool_version:       {build_info['tool_version']}")
    return 0


def _cmd_strategy_bench(args: argparse.Namespace) -> int:
    from hars_memory.eval.strategy_bench import StrategyBenchmarkError, run_strategy_matrix
    from hars_memory.strategies import StrategyConfigurationError, load_strategy_matrix

    try:
        matrix = load_strategy_matrix(args.config)
        report, report_path = run_strategy_matrix(matrix, run_id=args.run_id)
    except (StrategyConfigurationError, StrategyBenchmarkError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote strategy benchmark: {report_path}")
    print(f"matrix_sha256={report['matrix_sha256']}")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    """Query the LightRAG knowledge graph via CLI."""
    import asyncio

    from hars_memory.mcp_server import (
        DEFAULT_QUERY_TOP_K,
        _compute_hybrid_block,
        _get_rag,
        _lightrag_mode,
        _resolve_fetch_top_k,
        _resolve_query_mode,
        _staleness_info,
        STALE_INDEX_WARNING_DAYS,
    )

    question = args.question
    mode = args.mode or "hybrid"
    rag_mode = _lightrag_mode(mode)
    top_k = args.top_k or DEFAULT_QUERY_TOP_K
    ll_keywords = list(args.ll_keywords) if args.ll_keywords else []
    hl_keywords = list(args.hl_keywords) if args.hl_keywords else []
    kw_args = {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords} if (ll_keywords or hl_keywords) else {}
    rag_mode, mode_fallback = _resolve_query_mode(rag_mode, ll_keywords, hl_keywords)

    async def _run() -> dict:
        rag = await _get_rag()
        hybrid_block = await _compute_hybrid_block(rag, question, top_k, ll_keywords)
        if args.context_only:
            result = await rag.aquery(question, mode=rag_mode, top_k=top_k, **kw_args)
        else:
            result = await rag.aquery_llm(question, mode=rag_mode, top_k=top_k, **kw_args)
        return {
            "result": result,
            "hybrid": hybrid_block,
            "mode": rag_mode,
            "mode_fallback": mode_fallback,
        }

    try:
        data = asyncio.run(_run())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    last_ingest, stale_days = _staleness_info()
    staleness_warning = None
    if stale_days is not None and stale_days > STALE_INDEX_WARNING_DAYS:
        staleness_warning = (
            f"Knowledge graph last ingested {last_ingest} ({stale_days} days ago); "
            "anything added or changed after that date is absent from this response."
        )

    if args.json:
        payload = {
            "ok": True,
            "mode": data["mode"],
            "mode_fallback": data["mode_fallback"],
            "question": question,
            "top_k": top_k,
            "last_ingest": last_ingest,
            "stale_days": stale_days,
            "staleness_warning": staleness_warning,
            "hybrid": data["hybrid"],
            "result": str(data["result"]) if not isinstance(data["result"], dict) else data["result"],
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"mode: {data['mode']}")
        if data["mode_fallback"]:
            print(f"mode_fallback: {data['mode_fallback']}")
        if staleness_warning:
            print(f"WARNING: {staleness_warning}")
        hybrid = data["hybrid"]
        if hybrid.get("enabled", True):
            chunks = hybrid.get("fused_chunks", [])
            print(f"hybrid: {len(chunks)} fused chunks, {len(hybrid.get('identifier_matches', []))} identifier matches")
        raw = data["result"]
        if isinstance(raw, str):
            print(raw)
        elif isinstance(raw, dict):
            print(raw.get("response", json.dumps(raw, default=str)))
        else:
            print(str(raw))

    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Start the MCP server (stdio)."""
    mcp_main()
    return 0


def _cmd_server(args: argparse.Namespace) -> int:
    """Start the HTTP index-job service."""
    host = args.host or "127.0.0.1"
    port = args.port or 8787
    import os

    os.environ.setdefault("HARS_MEMORY_SERVICE_HOST", host)
    os.environ.setdefault("HARS_MEMORY_SERVICE_PORT", str(port))
    from hars_memory.service.server import main as server_main

    server_main()
    return 0


def _cmd_grpc(args: argparse.Namespace) -> int:
    """Start the gRPC server."""
    host = args.host
    port = args.port
    if host:
        import os as _os

        _os.environ["HARS_MEMORY_GRPC_HOST"] = host
    if port:
        import os as _os

        _os.environ["HARS_MEMORY_GRPC_PORT"] = str(port)
    grpc_main()
    return 0


def _cmd_consolidate(args: argparse.Namespace) -> int:
    """Trigger incremental reindex."""
    if args.paths:
        import os as _os

        _os.environ["HARS_MEMORY_INDEX_PATHS"] = " ".join(args.paths)
    if args.dry_run:
        import os as _os

        _os.environ["HARS_MEMORY_INDEX_DRY_RUN"] = "1"
    index_main()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Unified CLI for hars-longterm-memory: query, servers, corpus management, evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build (or incrementally re-index) a corpus index.")
    build_p.add_argument("--paths", nargs="+", required=True, type=Path)
    build_p.add_argument("--index-dir", required=True, type=Path)
    build_p.add_argument("--chunk-size", type=int, default=None)
    build_p.add_argument("--chunk-overlap", type=int, default=None)
    build_p.add_argument(
        "--force", action="store_true",
        help="Overwrite a non-empty --index-dir that was not itself produced by a prior build.",
    )
    build_p.set_defaults(func=_cmd_build)

    query_p = sub.add_parser("query", help="Search a corpus-built index.")
    query_p.add_argument("--index-dir", required=True, type=Path)
    query_p.add_argument("--question", required=True)
    query_p.add_argument("--top-k", type=int, default=None)
    query_p.add_argument("--mode", choices=("sparse", "dense", "fusion"), default=None)
    query_p.set_defaults(func=_cmd_query)

    eval_p = sub.add_parser("eval", help="Run retrieval metrics against a corpus-built index.")
    eval_p.add_argument("--index-dir", required=True, type=Path)
    eval_p.add_argument("--queries", required=True, type=Path)
    eval_p.add_argument("--report", required=True, type=Path)
    eval_p.add_argument("--mode", choices=("sparse", "dense", "fusion"), default=None)
    eval_p.add_argument("--top-k", type=int, default=None)
    eval_p.set_defaults(func=_cmd_eval)

    regress_p = sub.add_parser(
        "regress", help="Compare two eval reports; exit 1 on regression (CI gate)."
    )
    regress_p.add_argument("--baseline", required=True, type=Path)
    regress_p.add_argument("--candidate", required=True, type=Path)
    regress_p.add_argument("--report", type=Path, default=None)
    regress_p.add_argument(
        "--baseline-ab-bench-config",
        default=None,
        help=(
            "Which config to compare from the baseline report, when it's an "
            "ab_bench.py report with more than one config (required unless "
            "the report only ever ran one config)."
        ),
    )
    regress_p.add_argument(
        "--candidate-ab-bench-config",
        default=None,
        help="Same as --baseline-ab-bench-config, for the candidate report.",
    )
    regress_p.set_defaults(func=_cmd_regress)

    status_p = sub.add_parser("status", help="Print a corpus manifest summary.")
    status_p.add_argument("--index-dir", required=True, type=Path)
    status_p.set_defaults(func=_cmd_status)

    strategy_p = sub.add_parser(
        "strategy-bench",
        help="Build and evaluate a declarative index x search strategy matrix.",
    )
    strategy_p.add_argument("--config", required=True, type=Path)
    strategy_p.add_argument("--run-id", default=None)
    strategy_p.set_defaults(func=_cmd_strategy_bench)

    # -- Servers --
    recall_p = sub.add_parser("recall", help="Query the LightRAG knowledge graph.")
    recall_p.add_argument("question", help="Natural language query")
    recall_p.add_argument("--mode", choices=("local", "global", "hybrid", "naive"), default="hybrid")
    recall_p.add_argument("--top-k", type=int, default=20)
    recall_p.add_argument("--context-only", action="store_true", default=True)
    recall_p.add_argument("--json", action="store_true", help="Output raw JSON")
    recall_p.add_argument("--ll-keywords", nargs="*", default=None)
    recall_p.add_argument("--hl-keywords", nargs="*", default=None)
    recall_p.set_defaults(func=_cmd_recall)

    mcp_p = sub.add_parser("mcp", help="Start the MCP server (stdio).")
    mcp_p.set_defaults(func=_cmd_mcp)

    server_p = sub.add_parser("server", help="Start the HTTP index-job service.")
    server_p.add_argument("--host", default="127.0.0.1")
    server_p.add_argument("--port", type=int, default=8787)
    server_p.set_defaults(func=_cmd_server)

    grpc_p = sub.add_parser("grpc", help="Start the gRPC server.")
    grpc_p.add_argument("--host", default=None)
    grpc_p.add_argument("--port", type=int, default=None)
    grpc_p.set_defaults(func=_cmd_grpc)

    consolidate_p = sub.add_parser("consolidate", help="Trigger incremental reindex.")
    consolidate_p.add_argument("--paths", nargs="*", default=[".plans", "docs"])
    consolidate_p.add_argument("--dry-run", action="store_true", default=True)
    consolidate_p.set_defaults(func=_cmd_consolidate)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_project_root()
    parser = _build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)
