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
failures of the *insertion call itself* -- ``rag.initialize_storages()``
raising (e.g. a broken vector store), ``rag.adelete_by_doc_id()`` raising, or
``rag.ainsert()`` raising a programming error -- not per-document extraction
failures, which are LightRAG's own concern to retry.

Content fingerprinting does NOT gate ``ainsert()``
----------------------------------------------------
Every call to ``ingest_documents()`` hands *every* incoming document to
``ainsert()``, exactly like ``server/index.py``'s walk-driven CLI does for
its own full document set on every run. The content fingerprint (see
``change_detection.py``) is used only to detect documents whose content
changed since a previous call, so their stale content can be force-deleted
(``adelete_by_doc_id``) before the batched reinsert -- LightRAG's own
doc_id-based dedupe would otherwise silently keep the old content forever.
Unchanged documents are still passed to ``ainsert()`` on every call (this is
cheap: LightRAG's own doc_status set-diff, done at enqueue time before any
embed/extract work, already skips already-``processed`` doc_ids and retries
``FAILED``/``PENDING``/``PROCESSING`` ones). An earlier version of this
function used the fingerprint to *skip* calling ``ainsert()`` for
byte-identical content -- that defeated the self-healing story above, since
a document LightRAG had marked ``FAILED`` would never be handed to
``ainsert()`` again as long as its content stayed unchanged. Fingerprints
are only persisted (``store.save()``) after the whole insert call completes
without raising, so a failed/errored call leaves the on-disk store exactly
as it was and is safe to retry with the same input.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    default_fingerprint_store_path,
    detect_changed_documents,
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
    """Outcome of an `ingest_documents()` call.

    `documents_skipped` is informational only: it counts documents whose
    content fingerprint was unchanged from a previous call. It does NOT mean
    those documents were excluded from `ainsert()` -- every document in the
    call is always passed to `ainsert()`; see the module docstring.
    """

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
    ``server/index.py --refresh-changed``: every document is handed to
    LightRAG's ``ainsert()`` on every call (see this module's docstring,
    "Content fingerprinting does NOT gate ``ainsert()``"); a document whose
    content changed since a previous call is additionally deleted (by
    ``doc_id``) before that same batched insert. ``documents_skipped``
    reports how many of the incoming documents were byte-identical to what
    was recorded for their ``doc_id`` on a previous call -- informational
    only, it does not mean they were excluded from ``ainsert()``. See this
    module's docstring for what "written" does and does not guarantee about
    downstream LLM extraction.
    """
    return asyncio.run(_ingest_documents_async(docs, index_dir))


async def _ingest_documents_async(docs: list[Document], index_dir: Path) -> IngestResult:
    if not docs:
        return IngestResult(documents_written=0, documents_skipped=0, errors=[])

    fingerprint_path = default_fingerprint_store_path(index_dir)
    store = FingerprintStore.load(fingerprint_path)
    report = detect_changed_documents(docs, store)

    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag(working_dir=str(index_dir))
    written = 0
    errors: list[str] = []
    initialized = False
    try:
        await rag.initialize_storages()
        initialized = True
        for doc_id in report.changed_doc_ids:
            # Content changed since the last recorded fingerprint for this
            # doc_id -- delete the stale version before reinsert so
            # LightRAG's doc_id-based dedupe cannot silently keep it.
            await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]
        batch_size = int(
            os.environ.get(_INSERT_BATCH_SIZE_ENV, str(_DEFAULT_INSERT_BATCH_SIZE))
        )
        await _insert_all_batches(rag, docs, batch_size)
        written = len(docs)
        # Only persist fingerprints once the delete+insert above is fully
        # done -- an exception anywhere in this block leaves the on-disk
        # store untouched, so a retried call sees the same previous/current
        # comparison it would have seen if this call had never happened.
        store.save()
    except Exception as exc:  # noqa: BLE001 -- reported to the caller, not swallowed
        errors.append(str(exc))
    finally:
        if initialized:
            await rag.finalize_storages()

    return IngestResult(
        documents_written=written,
        documents_skipped=report.unchanged_count,
        errors=errors,
    )


__all__ = ["IngestResult", "ingest_documents"]
