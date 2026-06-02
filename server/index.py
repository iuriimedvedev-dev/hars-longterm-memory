#!/usr/bin/env python3
"""Full indexing entrypoint — run this ONLY when the GPU is free.

GPU GUARD: this script refuses to start if a vea/expert/ai_tuner/finetune/
distillation experiment is currently running on the backend.  It re-uses the
exact same concurrency-guard pattern as .claude/skills/distillation.py.

Usage (run when GPU is free):
    python tools/graphrag/server/index.py \\
        --paths .reports .plans .session \\
        --db-export \\
        [--full]          # full reindex (ignore change detection)
        [--dry-run]       # walk + count only, no LLM calls

Environment variables (see config/.env.example):
    GRAPHRAG_EXTRACTOR_BASE_URL  - llama-server endpoint for Qwen3.6-27B
    GRAPHRAG_EXTRACTOR_MODEL     - model name
    GRAPHRAG_WORKING_DIR         - LightRAG KV store dir
    GRAPHRAG_QDRANT_URL          - Qdrant service
    GRAPHRAG_POSTGRES_DSN        - hars-postgres DSN (read-only)
    HARS_API_BASE_URL            - HARS backend (for GPU guard)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure tools/ is on the path when run as a script from project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.graphrag.ingest.chunker import chunk_text
from tools.graphrag.ingest.walker import walk
from tools.graphrag.server.gpu_guard import GpuBusyError, assert_gpu_free

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graphrag.index")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build / update the HARS GraphRAG knowledge graph index."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[".reports", ".plans", ".session"],
        help="Directories or files to ingest (relative to project root).",
    )
    parser.add_argument(
        "--db-export",
        action="store_true",
        default=False,
        help="Also export experiments/hypotheses from hars-postgres.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Force full reindex (ignore change detection).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Walk + count documents only — no LLM extraction, no DB writes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="HARS backend URL for GPU guard (overrides HARS_API_BASE_URL env).",
    )
    return parser.parse_args()


async def _run_indexing(args: argparse.Namespace) -> None:
    api_url = args.api_base_url or os.environ.get("HARS_API_BASE_URL", "http://localhost:8765")

    # -----------------------------------------------------------------------
    # GPU guard — refuse to run while training is active.
    # -----------------------------------------------------------------------
    if not args.dry_run:
        try:
            assert_gpu_free(api_url)
        except GpuBusyError as exc:
            logger.error("BLOCKED: %s", exc)
            logger.error(
                "Wait for the training run to complete, then re-run:\n"
                "  python tools/graphrag/server/index.py --paths .reports .plans --db-export"
            )
            sys.exit(1)

    project_root = _PROJECT_ROOT
    resolved_paths = [
        project_root / p if not Path(p).is_absolute() else Path(p)
        for p in args.paths
    ]

    # -----------------------------------------------------------------------
    # Directory walk
    # -----------------------------------------------------------------------
    logger.info("Walking paths: %s", [str(p) for p in resolved_paths])
    docs, stats = walk(resolved_paths, dry_run=args.dry_run)
    logger.info(
        "Walk result: %d documents accepted (%s)",
        stats.files_accepted,
        json.dumps(stats.per_kind, indent=None),
    )

    # -----------------------------------------------------------------------
    # Postgres export
    # -----------------------------------------------------------------------
    db_docs: list = []
    if args.db_export:
        dsn = os.environ.get("GRAPHRAG_POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/hars")
        logger.info("Exporting Postgres tables from: %s", dsn.split("@")[-1])
        try:
            from tools.graphrag.ingest.postgres_export import export_all
            db_docs, db_stats = await export_all(dsn)
            logger.info(
                "Postgres export: %d experiments, %d hypotheses, %d links",
                db_stats.experiments,
                db_stats.hypotheses,
                db_stats.hypothesis_links,
            )
        except Exception as exc:
            logger.warning("Postgres export failed (skipping): %s", exc)

    all_docs = docs + db_docs
    logger.info("Total documents to index: %d", len(all_docs))

    # -----------------------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------------------
    all_chunks: list = []
    for doc in all_docs:
        chunks = chunk_text(
            doc.content,
            source_id=doc.doc_id,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))

    if args.dry_run:
        logger.info("DRY RUN complete — no LLM extraction performed.")
        logger.info(
            "Summary:\n  documents=%d\n  chunks=%d\n  per_kind=%s",
            len(all_docs),
            len(all_chunks),
            json.dumps(stats.per_kind, indent=2),
        )
        return

    # -----------------------------------------------------------------------
    # LightRAG insertion (requires GPU-free extractor LLM)
    # -----------------------------------------------------------------------
    from tools.graphrag.server.lightrag_init import create_lightrag
    from tools.graphrag.schema.extraction_prompt import EXTRACTION_SYSTEM_PROMPT

    rag = create_lightrag()
    logger.info("LightRAG instance ready, starting insertion...")

    # Batch insert — LightRAG's ainsert accepts a string or list of strings.
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        try:
            await rag.ainsert(texts)
            logger.info("Inserted batch %d/%d", i // batch_size + 1, (len(all_chunks) + batch_size - 1) // batch_size)
        except Exception as exc:
            logger.error("Insertion error on batch %d: %s", i // batch_size + 1, exc)
            raise

    logger.info("Indexing complete.")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_indexing(args))


if __name__ == "__main__":
    main()
