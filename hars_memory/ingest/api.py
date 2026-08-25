"""Public ingest API for hars-longterm-memory.

External callers (in-repo scripts, or another project entirely) that have
their own document source -- a database export, an API pull, anything that
isn't a plain file on disk -- construct `Document` objects themselves and
hand them to `ingest_documents()`. This is the one supported integration
point for non-file-walk ingestion; it is what keeps hars_memory itself free
of any particular external system's schema or vocabulary.

Pipeline reuse
--------------
This module does NOT duplicate the chunk -> embed -> extract -> insert
pipeline. It builds a LightRAG instance the same way the CLI entrypoint
(``hars_memory.server.index``) does, via
``hars_memory.server.lightrag_init.create_lightrag()``, and hands the
documents to that same module's ``_insert_all_batches()`` helper -- the
exact function ``server/index.py``'s ``--paths``/``--db-export`` walk-driven
CLI already uses to call ``rag.ainsert()`` in batches. ``ingest_documents()``
is a thin wrapper parameterised by an explicit ``list[Document]`` instead of
a file-walk result.

"documents_written" semantics
------------------------------
A document counts as "written" once it has been hand off to LightRAG's
``ainsert()`` pipeline without that call raising -- i.e. its content and
chunks are durably enqueued in the index. It does NOT require that
downstream entity/relation extraction (the GPU-bound LLM call) succeeded:
LightRAG marks a document ``FAILED`` (with an error message, not a raised
exception) when extraction fails, e.g. because the extractor LLM endpoint is
unreachable, and automatically retries FAILED documents the next time the
enqueue/process pipeline runs (see
``LightRAG.apipeline_process_enqueue_documents``, which pulls
``PENDING``/``PROCESSING``/``FAILED`` docs on every call). Treating "written"
as "durably enqueued for indexing (with self-healing retry)" rather than
"fully graph-extracted this call" is what lets ``ingest_documents()`` be
called safely regardless of whether the extractor LLM happens to be up right
now -- exactly the GPU-optional posture the rest of this package's ingest
path (file walk, change detection) already has. ``errors`` therefore reports
failures of the *insertion call itself* (e.g. a broken vector store, a
programming error) -- not per-document extraction failures, which are
LightRAG's own concern to retry.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    compute_fingerprint,
    default_fingerprint_store_path,
)
from hars_memory.ingest.document import Document
from hars_memory.server.index import _insert_all_batches

# Docs-per-ainsert() batch size. Mirrors server/index.py's own
# HARS_MEMORY_INSERT_BATCH_SIZE env var/default so both entrypoints into the
# same insertion helper behave identically unless an operator overrides it.
_INSERT_BATCH_SIZE_ENV = "HARS_MEMORY_INSERT_BATCH_SIZE"
_DEFAULT_INSERT_BATCH_SIZE = 10


@dataclass(slots=True)
class IngestResult:
    """Outcome of an `ingest_documents()` call."""

    documents_written: int
    documents_skipped: int
    errors: list[str] = field(default_factory=list)


def ingest_documents(docs: list[Document], *, index_dir: Path) -> IngestResult:
    """Ingest pre-built `Document` objects into the index at `index_dir`.

    This is the integration point for any document source that isn't a
    plain file on disk (e.g. rows exported from an external database).
    File-based ingestion should continue to use the `memory build`/
    `memory-index` walker path instead -- this function does not walk
    anything, it only accepts documents the caller has already built.

    Uses the same content-fingerprint change detection as
    ``server/index.py --refresh-changed``: a document whose content is
    byte-identical to what was recorded for the same ``doc_id`` on a
    previous call is skipped (no re-embed/re-extract); a document whose
    content changed is deleted (by ``doc_id``) and reinserted; a never-seen
    ``doc_id`` is inserted fresh. See this module's docstring for what
    "written" does and does not guarantee about downstream LLM extraction.
    """
    return asyncio.run(_ingest_documents_async(docs, index_dir))


async def _ingest_documents_async(docs: list[Document], index_dir: Path) -> IngestResult:
    if not docs:
        return IngestResult(documents_written=0, documents_skipped=0, errors=[])

    fingerprint_path = default_fingerprint_store_path(index_dir)
    store = FingerprintStore.load(fingerprint_path)

    to_insert: list[Document] = []
    stale_doc_ids: list[str] = []
    skipped = 0
    for doc in docs:
        current = compute_fingerprint(doc.content)
        previous = store.get(doc.doc_id)
        if previous == current:
            skipped += 1
        else:
            if previous is not None:
                # Content changed since the last recorded fingerprint for
                # this doc_id -- delete the stale version before reinsert so
                # LightRAG's doc_id-based dedupe cannot silently keep it.
                stale_doc_ids.append(doc.doc_id)
            to_insert.append(doc)
        store.set(doc.doc_id, current)

    written = 0
    errors: list[str] = []
    if to_insert:
        from hars_memory.server.lightrag_init import create_lightrag

        rag = create_lightrag(working_dir=str(index_dir))
        await rag.initialize_storages()
        try:
            for doc_id in stale_doc_ids:
                await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]
            batch_size = int(
                os.environ.get(_INSERT_BATCH_SIZE_ENV, str(_DEFAULT_INSERT_BATCH_SIZE))
            )
            await _insert_all_batches(rag, to_insert, batch_size)
            written = len(to_insert)
        except Exception as exc:  # noqa: BLE001 -- reported to the caller, not swallowed
            errors.append(str(exc))
        finally:
            await rag.finalize_storages()

    store.save()
    return IngestResult(documents_written=written, documents_skipped=skipped, errors=errors)


__all__ = ["IngestResult", "ingest_documents"]
