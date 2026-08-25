from pathlib import Path

import pytest

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


def test_source_kind_has_no_postgres_specific_members() -> None:
    from hars_memory.ingest.document import SourceKind

    names = {member.name for member in SourceKind}
    assert "POSTGRES_EXPERIMENT" not in names
    assert "POSTGRES_HYPOTHESIS" not in names
    assert "POSTGRES_HYPOTHESIS_LINK" not in names
    assert "EXTERNAL" in names
