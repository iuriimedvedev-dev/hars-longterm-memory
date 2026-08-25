#!/usr/bin/env python3
"""Full indexing entrypoint.

GPU GUARD: this package no longer checks GPU concurrency itself — that check
moved to tools/memory-config/scripts/gpu_guard.py (Cortex-owned). Cortex's own
indexing wrapper (update_kb.sh) must call assert_gpu_free() before invoking
this CLI; running this script directly does not protect a shared GPU.

Usage:
    python tools/memory/server/index.py \\
        --paths .reports .plans .session \\
        [--full]              # full reindex (ignore change detection)
        [--refresh-changed]   # opt-in: delete+reinsert docs whose content
                               # changed since last ingest (default OFF, never
                               # deletes on its own)
        [--dry-run]            # walk + count only, no LLM calls

Environment variables (see config/.env.example):
    HARS_MEMORY_EXTRACTOR_BASE_URL         - llama-server endpoint for Qwen3.6-27B
    HARS_MEMORY_EXTRACTOR_MODEL            - model name
    HARS_MEMORY_INDEX_DIR                - LightRAG KV store dir
    HARS_MEMORY_VECTOR_STORAGE             - LightRAG vector backend
    HARS_MEMORY_CLAUDE_MEMORY_DIR                 - optional extra ingest root (e.g. the
                                            Claude Code project-memory dir)
    HARS_MEMORY_FINGERPRINT_STORE          - sidecar JSON path for
                                            --refresh-changed content
                                            fingerprints (default: alongside
                                            HARS_MEMORY_INDEX_DIR)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Final

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    default_fingerprint_store_path,
    detect_changed_documents,
)
from hars_memory.ingest.document import Document
from hars_memory.ingest.walker import walk
from hars_memory.server.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger("memory.index")

# Optional extra ingest root, e.g. the Claude Code project-memory directory
# (~90 curated .md knowledge files). Unset by default — each operator points
# this at their own path; never hardcoded since it lives outside the repo and
# is user/machine-specific.
_MEMORY_DIR_ENV: Final[str] = "HARS_MEMORY_CLAUDE_MEMORY_DIR"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build / update the HARS long-term memory knowledge graph index."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[".plans", "docs"],
        help="Directories or files to ingest (relative to project root).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Force full reindex (ignore change detection).",
    )
    parser.add_argument(
        "--refresh-changed",
        action="store_true",
        default=False,
        help=(
            "Opt-in, default OFF: detect files/rows whose CONTENT changed since "
            "the last ingest under the same doc_id (a file edited in place is "
            "otherwise skipped forever) and delete+reinsert only those. Compares "
            "a sha256 content fingerprint stored in a sidecar JSON "
            "(HARS_MEMORY_FINGERPRINT_STORE). Documents with no recorded "
            "fingerprint yet are NEVER deleted — only newly tracked. Without "
            "this flag, no delete is ever issued."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Walk + count documents only — no LLM extraction, no DB writes.",
    )
    return parser.parse_args()


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    insert_task_holder: list[asyncio.Task[None]],
) -> None:
    """Install SIGINT / SIGTERM handlers on *loop*.

    First signal: log, set *shutdown_event*, cancel the in-flight insert task.
    Second signal: immediate hard exit (os._exit) so the process never hangs.
    """
    signal_count = 0

    def _handle_signal(sig: signal.Signals) -> None:
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            logger.warning(
                "Graceful shutdown requested (%s): finishing in-flight work and "
                "flushing storages — partial index will be preserved. "
                "Send signal again to force-quit immediately.",
                sig.name,
            )
            shutdown_event.set()
            for task in insert_task_holder:
                if not task.done():
                    task.cancel()
        else:
            logger.error(
                "Second signal received (%s): forcing immediate exit. "
                "Some in-flight data may be lost.",
                sig.name,
            )
            os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)


def _resolve_ingest_paths(cli_paths: list[str], project_root: Path) -> list[Path]:
    """Resolve --paths CLI values to absolute Paths, plus the optional
    Claude memory directory (HARS_MEMORY_CLAUDE_MEMORY_DIR), if configured.

    The memory dir is appended unconditionally when the env var is set —
    independent of whatever --paths override the caller passed — since it is
    a standing ingest root (~90 curated knowledge files), not an ephemeral one.
    """
    resolved = [
        project_root / p if not Path(p).is_absolute() else Path(p)
        for p in cli_paths
    ]
    memory_dir_raw = os.environ.get(_MEMORY_DIR_ENV, "").strip()
    if memory_dir_raw:
        memory_dir = Path(memory_dir_raw)
        if memory_dir not in resolved:
            resolved.append(memory_dir)
            logger.info(
                "Including Claude memory directory (%s=%s)", _MEMORY_DIR_ENV, memory_dir
            )
    return resolved


async def _apply_refresh_changed(
    rag: object,
    all_docs: list[Document],
    *,
    refresh_changed: bool,
    fingerprint_store_path: Path,
) -> None:
    """Detect content-changed documents and delete+reinsert them — IF opted in.

    Safety contract: when *refresh_changed* is False (the default), this is a
    strict no-op — the fingerprint sidecar is never read or written, and
    ``rag.adelete_by_doc_id`` is never called. A normal `index.py` run can
    never trigger a delete via this path.
    """
    if not refresh_changed:
        logger.info(
            "--refresh-changed not set: skipping content-change detection "
            "(no deletes issued)."
        )
        return

    store = FingerprintStore.load(fingerprint_store_path)
    report = detect_changed_documents(all_docs, store)
    logger.info(
        "Fingerprint check (%s): %d changed, %d unchanged, %d with no stored "
        "fingerprint (left untouched, now recorded)",
        fingerprint_store_path,
        len(report.changed_doc_ids),
        report.unchanged_count,
        report.no_fingerprint_count,
    )
    for doc_id in report.changed_doc_ids:
        logger.info("Deleting stale content for doc_id=%s before reinsert", doc_id)
        await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]
    store.save()


async def _insert_all_batches(
    rag: object,
    all_docs: list,
    batch_size: int,
) -> None:
    """Insert *all_docs* into *rag* in batches of *batch_size*.

    Designed to be run as a cancellable asyncio Task.  A CancelledError propagates
    naturally so the caller's finally block can still flush storages.
    """
    total_batches = (len(all_docs) + batch_size - 1) // batch_size
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        texts = [doc.content for doc in batch]
        ids = [doc.doc_id for doc in batch]
        file_paths = [doc.source_path for doc in batch]
        await rag.ainsert(texts, ids=ids, file_paths=file_paths)  # type: ignore[attr-defined]
        logger.info("Inserted batch %d/%d", i // batch_size + 1, total_batches)


async def _run_indexing(args: argparse.Namespace) -> None:
    # GPU-guard moved to tools/memory-config/scripts/gpu_guard.py — Cortex's own
    # indexing wrapper (update_kb.sh) must call assert_gpu_free() before invoking
    # this CLI; the package itself no longer knows Cortex's GPU is a shared
    # resource. (Task 4.x's cutover updates update_kb.sh to make this call.)

    project_root = _PROJECT_ROOT
    resolved_paths = _resolve_ingest_paths(args.paths, project_root)

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

    # Postgres export moved to tools/memory-config/scripts/postgres_export.py
    # (Cortex-specific, not part of the generic package — see
    # docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md §1).
    # Run that script separately (it calls hars_memory.ingest.api.ingest_documents()
    # directly) instead of via this walker-driven CLI.
    all_docs = docs
    logger.info("Total documents to index: %d", len(all_docs))

    if args.dry_run:
        logger.info("DRY RUN complete — no LLM extraction performed.")
        logger.info(
            "Summary:\n  documents=%d\n  per_kind=%s",
            len(all_docs),
            json.dumps(stats.per_kind, indent=2),
        )
        return

    # -----------------------------------------------------------------------
    # LightRAG insertion (requires GPU-free extractor LLM)
    # -----------------------------------------------------------------------
    from hars_memory.server.lightrag_init import create_lightrag, resolve_working_dir

    rag = create_lightrag()
    await rag.initialize_storages()
    logger.info("LightRAG instance ready, starting insertion...")

    # -----------------------------------------------------------------------
    # Content-change detection (opt-in via --refresh-changed; no-op otherwise)
    # -----------------------------------------------------------------------
    fingerprint_store_path = Path(
        os.environ.get(
            "HARS_MEMORY_FINGERPRINT_STORE",
            str(default_fingerprint_store_path(resolve_working_dir())),
        )
    )
    await _apply_refresh_changed(
        rag,
        all_docs,
        refresh_changed=args.refresh_changed,
        fingerprint_store_path=fingerprint_store_path,
    )

    # Signal-handling state — shared between the handler closure and this coroutine.
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    insert_task_holder: list[asyncio.Task[None]] = []  # populated just before install

    # Batch size = docs per ainsert() call.  LightRAG only runs max_parallel_insert
    # docs concurrently WITHIN one call and drains fully between calls, so a small
    # batch head-of-line-blocks all slots behind the largest doc in the batch.
    batch_size = int(os.environ.get("HARS_MEMORY_INSERT_BATCH_SIZE", "10"))
    interrupted = False
    insert_task: asyncio.Task[None] = loop.create_task(
        _insert_all_batches(rag, all_docs, batch_size)
    )
    insert_task_holder.append(insert_task)

    # Install handlers AFTER the task exists so the holder is populated.
    _install_signal_handlers(loop, shutdown_event, insert_task_holder)

    try:
        await insert_task
    except asyncio.CancelledError:
        interrupted = True
        logger.warning(
            "Insert task cancelled (signal received). "
            "Flushing partial index to disk — resume will skip already-processed documents."
        )
    except Exception as exc:
        logger.error("Insertion error: %s", exc)
        raise
    finally:
        try:
            await rag.finalize_storages()
            logger.info("LightRAG storages flushed to disk successfully.")
        except Exception as exc:
            logger.warning("LightRAG storage finalization failed: %s", exc)

    if interrupted:
        logger.info(
            "Graceful shutdown complete. "
            "Partial index preserved — re-run to resume from where indexing stopped."
        )
        sys.exit(130)

    logger.info("Indexing complete.")


def main() -> None:
    # Permanent fail-closed guard: refuse to start against a stale GRAPHRAG_*
    # env (pre-2026-07-29 rename) instead of silently falling back to
    # HARS_MEMORY_* defaults. See server/legacy_env_guard.py.
    from hars_memory.server.legacy_env_guard import refuse_if_legacy_graphrag_env

    refuse_if_legacy_graphrag_env()

    args = _parse_args()
    asyncio.run(_run_indexing(args))


if __name__ == "__main__":
    main()
