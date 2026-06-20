"""Postgres source exporter — reads experiments/hypotheses/hypothesis_links (read-only).

One stable-ID document per row.  No LLM calls.  GPU-free.

Usage (dry-run, no DB needed):
    from tools.graphrag.ingest.postgres_export import transform_experiment_row
    doc = transform_experiment_row({"id": "abc", "name": "test", "status": "running"})

Usage (live, requires DB):
    from tools.graphrag.ingest.postgres_export import export_all
    docs, stats = await export_all(dsn="postgresql://...")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from tools.graphrag.ingest.document import Document, SourceKind
from tools.graphrag.schema.entity_types import make_stable_id

logger = logging.getLogger(__name__)


@dataclass
class ExportStats:
    experiments: int = 0
    hypotheses: int = 0
    hypothesis_links: int = 0

    @property
    def total(self) -> int:
        return self.experiments + self.hypotheses + self.hypothesis_links


# ---------------------------------------------------------------------------
# Pure transform functions (testable without a DB connection)
# ---------------------------------------------------------------------------


def transform_experiment_row(row: dict[str, object]) -> Document:
    """Convert one experiments row to a stable-ID Document."""
    row_id = str(row.get("id", "unknown"))
    doc_id = make_stable_id("experiment", row_id)

    name = row.get("name") or row.get("workflow_type") or row_id
    status = row.get("status", "unknown")
    workflow = row.get("workflow_type", "")
    operation = row.get("operation", "")
    checkpoint_path = row.get("checkpoint_path", "")
    config_snippet = json.dumps(row.get("config") or {}, indent=2)[:800]
    metrics_snippet = json.dumps(row.get("final_metrics") or {}, indent=2)[:800]

    content = (
        f"Experiment ID: {row_id}\n"
        f"Name: {name}\n"
        f"Workflow: {workflow}\n"
        f"Operation: {operation}\n"
        f"Status: {status}\n"
        f"Checkpoint path: {checkpoint_path}\n"
        f"Final metrics (excerpt):\n{metrics_snippet}\n"
        f"Config (excerpt):\n{config_snippet}\n"
    )
    for extra_key in ("project", "job_id", "pipeline_id", "tags", "error", "baseline_experiment_id"):
        value = row.get(extra_key)
        if value:
            content += f"{extra_key.capitalize()}: {value}\n"

    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.POSTGRES_EXPERIMENT,
        source_path=f"postgres://experiments/{row_id}",
        metadata={k: v for k, v in row.items() if k not in ("config",)},
    )


def transform_hypothesis_row(row: dict[str, object]) -> Document:
    """Convert one hypotheses row to a stable-ID Document."""
    row_id = str(row.get("id", "unknown"))
    doc_id = make_stable_id("hypothesis", row_id)

    slug = row.get("slug", "")
    title = row.get("title", row_id)
    status = row.get("status", "unknown")
    description = row.get("description", "")
    rationale = row.get("rationale", "")
    success_criteria = row.get("success_criteria", "")
    tags = row.get("tags") or []
    tags_str = ", ".join(str(t) for t in tags) if tags else ""

    content = (
        f"Hypothesis ID: {row_id} (stable-id: {doc_id})\n"
        f"Slug: {slug}\n"
        f"Title: {title}\n"
        f"Status: {status}\n"
    )
    if description:
        content += f"Description: {description}\n"
    if rationale:
        content += f"Rationale: {rationale}\n"
    if success_criteria:
        content += f"Success criteria: {success_criteria}\n"
    if tags_str:
        content += f"Tags: {tags_str}\n"

    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.POSTGRES_HYPOTHESIS,
        source_path=f"postgres://hypotheses/{row_id}",
        metadata=dict(row),
    )


def transform_hypothesis_link_row(row: dict[str, object]) -> Document:
    """Convert one hypothesis_links row to a stable-ID Document."""
    row_id = str(row.get("id", "unknown"))
    hyp_id = str(row.get("hypothesis_id", ""))
    entity_type = str(row.get("entity_type", ""))
    entity_id = str(row.get("entity_id", ""))
    relation = str(row.get("relation_type", "related"))
    doc_id = f"link:{row_id}"

    content = (
        f"Hypothesis link ID: {row_id}\n"
        f"Hypothesis: hyp:{hyp_id}\n"
        f"Links to: {entity_type} / {entity_id}\n"
        f"Relation: {relation}\n"
    )
    if row.get("label"):
        content += f"Label: {row['label']}\n"
    if row.get("notes"):
        content += f"Notes: {row['notes']}\n"

    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.POSTGRES_HYPOTHESIS_LINK,
        source_path=f"postgres://hypothesis_links/{row_id}",
        metadata=dict(row),
    )


# ---------------------------------------------------------------------------
# Live DB export (requires asyncpg or psycopg2)
# ---------------------------------------------------------------------------

_EXPORT_QUERIES: dict[str, str] = {
    "experiments": (
        "SELECT id, project, name, operation, job_id, pipeline_id, config, tags, "
        "status, checkpoint_path, final_metrics, created_at, updated_at, "
        "workflow_type, error, baseline_experiment_id "
        "FROM experiments ORDER BY created_at DESC"
    ),
    "hypotheses": (
        "SELECT id, slug, title, status, description, rationale, "
        "success_criteria, tags, created_at, updated_at FROM hypotheses"
    ),
    "hypothesis_links": (
        "SELECT id, hypothesis_id, entity_type, entity_id, relation_type, "
        "label, notes, created_at FROM hypothesis_links"
    ),
}


async def export_all(dsn: str) -> tuple[list[Document], ExportStats]:
    """Export all three tables as Documents.

    Parameters
    ----------
    dsn:
        PostgreSQL DSN, e.g. ``postgresql://host:5432/db``. Credentials must
        come from the environment or a secret manager, not committed defaults.

    Returns
    -------
    (documents, stats):
        All documents + per-table row counts.

    Raises
    ------
    ImportError
        If neither ``asyncpg`` nor ``psycopg2`` is installed.
    RuntimeError
        If the DB connection fails.
    """
    try:
        import asyncpg  # type: ignore[import-not-found]
        return await _export_asyncpg(dsn)
    except ImportError:
        pass
    try:
        import psycopg2  # type: ignore[import-not-found]
        import psycopg2.extras
        return _export_psycopg2(dsn, psycopg2, psycopg2.extras)
    except ImportError as exc:
        raise ImportError(
            "postgres_export requires asyncpg or psycopg2. "
            "Install with: pip install asyncpg"
        ) from exc


async def _export_asyncpg(dsn: str) -> tuple[list[Document], ExportStats]:
    import asyncpg  # type: ignore[import-not-found]

    docs: list[Document] = []
    stats = ExportStats()

    conn = await asyncpg.connect(dsn)
    try:
        for row in await conn.fetch(_EXPORT_QUERIES["experiments"]):
            docs.append(transform_experiment_row(dict(row)))
            stats.experiments += 1

        for row in await conn.fetch(_EXPORT_QUERIES["hypotheses"]):
            docs.append(transform_hypothesis_row(dict(row)))
            stats.hypotheses += 1

        for row in await conn.fetch(_EXPORT_QUERIES["hypothesis_links"]):
            docs.append(transform_hypothesis_link_row(dict(row)))
            stats.hypothesis_links += 1
    finally:
        await conn.close()

    logger.info(
        "Postgres export: %d experiments, %d hypotheses, %d links",
        stats.experiments,
        stats.hypotheses,
        stats.hypothesis_links,
    )
    return docs, stats


def _export_psycopg2(
    dsn: str,
    psycopg2: object,
    extras: object,
) -> tuple[list[Document], ExportStats]:
    docs: list[Document] = []
    stats = ExportStats()

    conn = psycopg2.connect(dsn)  # type: ignore[attr-defined]
    try:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)  # type: ignore[attr-defined]

        cursor.execute(_EXPORT_QUERIES["experiments"])
        for row in cursor.fetchall():
            docs.append(transform_experiment_row(dict(row)))
            stats.experiments += 1

        cursor.execute(_EXPORT_QUERIES["hypotheses"])
        for row in cursor.fetchall():
            docs.append(transform_hypothesis_row(dict(row)))
            stats.hypotheses += 1

        cursor.execute(_EXPORT_QUERIES["hypothesis_links"])
        for row in cursor.fetchall():
            docs.append(transform_hypothesis_link_row(dict(row)))
            stats.hypothesis_links += 1
    finally:
        conn.close()

    logger.info(
        "Postgres export (psycopg2): %d experiments, %d hypotheses, %d links",
        stats.experiments,
        stats.hypotheses,
        stats.hypothesis_links,
    )
    return docs, stats
