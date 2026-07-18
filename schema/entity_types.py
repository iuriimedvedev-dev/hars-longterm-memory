"""Domain entity and relation type definitions for the HARS GraphRAG knowledge graph.

These types seed the LightRAG extraction prompt.  They are also used by the
ingest layer when building stable-ID documents from Postgres rows.
"""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    """Typed entities extracted from HARS corpus.

    IMPORTANT: LightRAG rejects entity types that contain ``/``, ``|``, or
    other special characters (see lightrag/operate.py).  All values here must
    be slash-free.  The extractor LLM lowercases and strips spaces before
    storing, so names are matched case-insensitively at insertion time.

    ``Model/Backbone`` was split into ``Model`` + ``Backbone`` and
    ``Pipeline/Phase`` was split into ``Pipeline`` + ``Phase`` to ensure the
    LLM's output is accepted rather than silently dropped.
    """

    HYPOTHESIS = "Hypothesis"
    EXPERIMENT = "Experiment"
    JOB_RUN = "JobRun"
    CHECKPOINT = "Checkpoint"
    METRIC = "Metric"
    DATASET = "Dataset"
    MODEL = "Model"
    BACKBONE = "Backbone"
    ENCODER = "Encoder"
    PIPELINE = "Pipeline"
    PHASE = "Phase"
    REPORT = "Report"
    PLAN = "Plan"
    FINDING = "Finding"
    CONFIG = "Config"
    COMPONENT = "Component"
    # Added 2026-07-10: the extractor kept emitting these on real corpus chunks
    # and LightRAG dropped the entities as invalid-type. Keep the list tight.
    ARTIFACT = "Artifact"    # files, paths, tarballs, logs
    CONCEPT = "Concept"      # methods, ideas, failure modes
    TOOL = "Tool"            # CLIs, services, libraries
    TASK = "Task"            # work items, action points


class RelationType(str, Enum):
    """Typed relations between HARS entities."""

    TESTS = "tests"
    CONFIRMS = "confirms"
    INVALIDATES = "invalidates"
    DERIVED_FROM = "derived_from"
    PRODUCES = "produces"
    EVALUATED_ON = "evaluated_on"
    OUTPERFORMS = "outperforms"
    REGRESSES = "regresses"
    SUPERSEDES = "supersedes"
    USES = "uses"
    DOCUMENTED_IN = "documented_in"
    DEPENDS_ON = "depends_on"


# Stable-ID prefixes for DB-origin nodes (ensures prose mentions resolve to
# the same graph node as the Postgres row document).
ENTITY_ID_PREFIXES: dict[str, str] = {
    "hypothesis": "hyp",
    "experiment": "exp",
    "checkpoint": "ckpt",
    "job_run": "run",
}


def make_stable_id(source_table: str, row_id: str | int) -> str:
    """Return a stable, prefix-qualified entity ID for a DB row.

    Examples
    --------
    >>> make_stable_id("hypothesis", "abc-123")
    'hyp:abc-123'
    >>> make_stable_id("experiment", 42)
    'exp:42'
    """
    prefix = ENTITY_ID_PREFIXES.get(source_table, source_table[:3])
    return f"{prefix}:{row_id}"
