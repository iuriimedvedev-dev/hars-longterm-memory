"""Document model — the unit that flows from walker/postgres-exporter into LightRAG.

No LLM calls here.  This layer is fully GPU-free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SourceKind(str, Enum):
    MARKDOWN = "markdown"
    PYTHON = "python"
    JSON = "json"
    TEXT = "text"
    POSTGRES_EXPERIMENT = "postgres:experiment"
    POSTGRES_HYPOTHESIS = "postgres:hypothesis"
    POSTGRES_HYPOTHESIS_LINK = "postgres:hypothesis_link"


@dataclass(slots=True)
class Document:
    """A normalised unit of content ready for LightRAG ingestion.

    Attributes
    ----------
    doc_id:
        Stable, globally unique ID.  File-based documents use a path-hash
        prefix; Postgres rows use ``exp:<id>``, ``hyp:<id>``, etc.
    content:
        Full text content — either prose or a JSON-serialised record.
    source_kind:
        Enum tag identifying how this document was produced.
    source_path:
        Absolute path or a pseudo-path like ``postgres://experiments/42``.
    metadata:
        Arbitrary key-value pairs for provenance (mtime, row id, etc.).
    """

    doc_id: str
    content: str
    source_kind: SourceKind
    source_path: str
    metadata: dict[str, object] = field(default_factory=dict)


def file_stable_id(path: Path) -> str:
    """Derive a stable document ID from a file path.

    Uses a short SHA-256 prefix of the absolute path string so the ID
    survives directory renames while staying reproducible for the same file.

    >>> import re
    >>> doc_id = file_stable_id(Path("/some/path/report.md"))
    >>> bool(re.match(r"^file:[0-9a-f]{12}$", doc_id))
    True
    """
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return f"file:{digest}"


def infer_source_kind(path: Path) -> SourceKind:
    """Return the SourceKind for a given file extension."""
    suffix = path.suffix.lower()
    match suffix:
        case ".md":
            return SourceKind.MARKDOWN
        case ".py":
            return SourceKind.PYTHON
        case ".json":
            return SourceKind.JSON
        case ".txt":
            return SourceKind.TEXT
        case _:
            return SourceKind.TEXT
