"""Importable console entry points for the HARS long-term memory UV project.

``index_main`` / ``mcp_main`` / ``eval_battle_main`` are the pre-existing
entry points (``memory-index`` / ``memory-mcp`` / ``memory-eval-battle`` in
``pyproject.toml``) — unchanged, kept working exactly as before (``memory-mcp``
in particular: the MCP server is live and other agents depend on it).

``main`` is the new unified ``memory`` console script (``memory <subcommand>``),
covering the LLM-free ``tools/memory/corpus`` build/query/eval/regress
pipeline. It imports every heavier dependency (corpus/eval/regression
modules) lazily, INSIDE each subcommand handler, not at module scope — so
`memory --help`, `memory-mcp`, `memory-index` etc. never pay an import cost
for subsystems they don't use, mirroring the existing lazy-import pattern
already used by ``mcp_main`` below.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_project_root() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def index_main() -> None:
    _ensure_project_root()
    from tools.memory.server.index import main

    main()


def mcp_main() -> None:
    _ensure_project_root()
    script_path = _project_root() / "plugins" / "hars-longterm-memory" / "scripts" / "hars_longterm_memory_mcp.py"
    spec = importlib.util.spec_from_file_location("hars_longterm_memory_mcp", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MCP server module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    asyncio.run(module.main())


def eval_battle_main() -> None:
    _ensure_project_root()
    from tools.memory.eval.battle import main

    main()


# ---------------------------------------------------------------------------
# `memory` unified subcommand CLI (corpus build/query/eval/regress/status)
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    from tools.memory.corpus.build import (
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
    from tools.memory.corpus.query import DEFAULT_TOP_K, CorpusQueryError, search

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
    from tools.memory.eval.corpus_eval import DEFAULT_MODE, DEFAULT_TOP_K, CorpusEvalError, run_eval

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
    from tools.memory.eval.regression import RegressionError, compare_reports

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        verdict = compare_reports(baseline, candidate)
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
    from tools.memory.corpus.build import MANIFEST_FILENAME

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="LLM-free, CPU-only corpus build/query/eval subsystem for tools/memory.",
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
    regress_p.set_defaults(func=_cmd_regress)

    status_p = sub.add_parser("status", help="Print a corpus manifest summary.")
    status_p.add_argument("--index-dir", required=True, type=Path)
    status_p.set_defaults(func=_cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_project_root()
    parser = _build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)
