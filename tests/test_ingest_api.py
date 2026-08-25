from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from hars_memory.ingest import api as api_module
from hars_memory.ingest.api import IngestResult, ingest_documents
from hars_memory.ingest.document import Document, SourceKind


def test_ingest_documents_returns_result_with_counts(tmp_path: Path) -> None:
    docs = [
        Document(
            doc_id="ext:1",
            content="Experiment E1 confirmed hypothesis H3.",
            source_kind=SourceKind.EXTERNAL,
            source_path="postgres://experiments/1",
            metadata={"origin_table": "experiments", "row_id": "1"},
        ),
    ]
    result = ingest_documents(docs, index_dir=tmp_path)
    assert isinstance(result, IngestResult)
    assert result.documents_written == 1
    assert result.documents_skipped == 0
    assert result.errors == []


def _make_fake_rag() -> AsyncMock:
    rag = AsyncMock()
    rag.initialize_storages = AsyncMock(return_value=None)
    rag.finalize_storages = AsyncMock(return_value=None)
    rag.adelete_by_doc_id = AsyncMock(return_value=None)
    return rag


def test_failed_insert_does_not_permanently_skip_unchanged_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc must be retried on the next call even if its content is
    byte-identical to a previous call that errored out.

    This is the specific failure mode of the Critical review finding on
    commit 53067f1d: persisting the fingerprint unconditionally (before/
    regardless of insert success) meant a document that failed to insert
    once would be silently skipped forever on every subsequent call with
    unchanged content -- defeating LightRAG's own FAILED-doc retry story
    that this module's docstring relies on.
    """
    doc = Document(
        doc_id="ext:retry-1",
        content="Experiment E9 confirmed hypothesis H1.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/9",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    call_doc_id_batches: list[list[str]] = []

    async def failing_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])
        raise RuntimeError("simulated extractor-down insert failure")

    monkeypatch.setattr(api_module, "_insert_all_batches", failing_insert)

    first = ingest_documents([doc], index_dir=tmp_path)
    assert first.documents_written == 0
    assert first.documents_skipped == 0
    assert first.errors  # the failure is reported, not swallowed
    assert call_doc_id_batches == [["ext:retry-1"]]

    async def succeeding_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])

    monkeypatch.setattr(api_module, "_insert_all_batches", succeeding_insert)

    second = ingest_documents([doc], index_dir=tmp_path)
    assert second.errors == []
    # The critical assertion: even though content is byte-identical to the
    # first (failed) call, the doc must be handed to _insert_all_batches
    # again -- NOT silently skipped because a fingerprint match was recorded
    # despite the earlier failure.
    assert call_doc_id_batches == [["ext:retry-1"], ["ext:retry-1"]]
    assert second.documents_written == 1


def test_initialize_storages_failure_is_reported_via_errors_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken vector store (rag.initialize_storages() raising) must surface
    via IngestResult.errors, per the module docstring's own example -- not
    propagate as an unhandled exception out of ingest_documents()."""
    doc = Document(
        doc_id="ext:broken-store",
        content="Experiment E10 confirmed hypothesis H2.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/10",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    fake_rag.initialize_storages = AsyncMock(
        side_effect=RuntimeError("simulated corrupt vector store")
    )
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    insert_called = False

    async def unexpected_insert(rag, docs, batch_size):
        nonlocal insert_called
        insert_called = True

    monkeypatch.setattr(api_module, "_insert_all_batches", unexpected_insert)

    result = ingest_documents([doc], index_dir=tmp_path)

    assert result.documents_written == 0
    assert result.errors and "simulated corrupt vector store" in result.errors[0]
    assert insert_called is False
    # finalize_storages() must not be invoked on a rag whose storages never
    # finished initializing.
    fake_rag.finalize_storages.assert_not_called()


def test_source_kind_has_no_postgres_specific_members() -> None:
    from hars_memory.ingest.document import SourceKind

    names = {member.name for member in SourceKind}
    assert "POSTGRES_EXPERIMENT" not in names
    assert "POSTGRES_HYPOTHESIS" not in names
    assert "POSTGRES_HYPOTHESIS_LINK" not in names
    assert "EXTERNAL" in names
