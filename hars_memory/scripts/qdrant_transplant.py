#!/usr/bin/env python3
"""Bidirectional vector transplant between LightRAG's NanoVectorDBStorage
JSON files and QdrantVectorDBStorage — pure data copy, zero re-embedding.

Forward (default): reads ``vdb_{chunks,entities,relationships}.json`` from a
LightRAG working dir and upserts every record into the corresponding
``lightrag_vdb_*`` Qdrant collection under a single tenant (``workspace_id``).

Reverse (``--reverse``): scrolls the three Qdrant collections back into valid
``vdb_*.json`` files, matching NanoVectorDB's on-disk format byte-for-byte
(same ``embedding_dim``/``data``/``matrix`` keys, same per-record ``vector``
field encoding: float16 -> zlib -> base64). This is what makes rollback real
after Qdrant-mode indexing has resumed — see
``.plans/2026-07-29_graphrag-qdrant-migration.md`` §8.

Point ID and payload construction deliberately import the real
``compute_mdhash_id_for_qdrant`` from the installed ``lightrag.kg.qdrant_impl``
rather than reimplementing sha256+uuid by hand — the whole point of this
script is byte-exact compatibility with what LightRAG's own
``QdrantVectorDBStorage.upsert()`` would have written, so any hand-rolled
reimplementation would itself be a source of subtle drift.

Usage
-----
    # Forward: JSON -> Qdrant (all three namespaces)
    uv run --project tools/memory python tools/memory/scripts/qdrant_transplant.py \\
        --working-dir /home/user/.local/share/hars-graphrag/index_gemma_v4 \\
        --qdrant-url http://localhost:6335 \\
        --workspace hars_longterm_memory \\
        --collection-prefix hars_longterm_memory

    # Reverse: Qdrant -> JSON (rollback / export current state)
    uv run --project tools/memory python tools/memory/scripts/qdrant_transplant.py \\
        --reverse \\
        --working-dir /tmp/qdrant_export \\
        --qdrant-url http://localhost:6335 \\
        --workspace hars_longterm_memory \\
        --collection-prefix hars_longterm_memory

``--collection-prefix`` (or env ``HARS_MEMORY_QDRANT_COLLECTION_PREFIX``) is
required — it namespaces the fixed ``lightrag_vdb_{chunks,entities,
relationships}`` collection names so multiple projects can share one Qdrant
container without colliding on schema (embedding dimension is a
per-collection property, not per-workspace).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from hars_memory.server.embedder import qdrant_collection_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qdrant_transplant")

# Namespace -> (JSON filename, meta_fields to carry into the Qdrant payload).
# Matches lightrag/lightrag.py's per-namespace meta_fields exactly (verified
# 2026-07-29 investigation, tools/memory/server/lightrag_init.py's factory).
NAMESPACES: dict[str, tuple[str, tuple[str, ...]]] = {
    "chunks": ("vdb_chunks.json", ("full_doc_id", "content", "file_path")),
    "entities": ("vdb_entities.json", ("entity_name", "source_id", "content", "file_path")),
    "relationships": (
        "vdb_relationships.json",
        ("src_id", "tgt_id", "source_id", "content", "file_path"),
    ),
}

ID_FIELD = "id"
WORKSPACE_ID_FIELD = "workspace_id"
CREATED_AT_FIELD = "created_at"
EXPECTED_EMBEDDING_DIM = 768


def _qdrant_collection_name(namespace: str, prefix: str) -> str:
    names = qdrant_collection_names(prefix)
    return dict(zip(NAMESPACES.keys(), names))[namespace]


def _compute_point_id(record_id: str, workspace: str):
    """Delegate to the real LightRAG implementation for byte-exact IDs."""
    from lightrag.kg.qdrant_impl import compute_mdhash_id_for_qdrant

    return compute_mdhash_id_for_qdrant(record_id, prefix=workspace)


# ---------------------------------------------------------------------------
# Forward: JSON -> Qdrant
# ---------------------------------------------------------------------------


def _load_vdb_json(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    embedding_dim = doc["embedding_dim"]
    if embedding_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"{path}: embedding_dim={embedding_dim}, expected {EXPECTED_EMBEDDING_DIM} "
            "(unsloth/embeddinggemma-300m). Refusing to transplant a mismatched index."
        )
    data = doc["data"]
    matrix = np.frombuffer(base64.b64decode(doc["matrix"]), dtype=np.float32).reshape(
        -1, embedding_dim
    )
    if len(matrix) != len(data):
        raise ValueError(
            f"{path}: matrix has {len(matrix)} rows but data has {len(data)} records "
            "— positional mapping broken, refusing to transplant."
        )
    return data, matrix


def transplant_namespace_forward(
    *,
    client: Any,
    working_dir: Path,
    namespace: str,
    workspace: str,
    collection_prefix: str,
    batch_size: int,
) -> dict[str, Any]:
    from qdrant_client import models

    filename, meta_fields = NAMESPACES[namespace]
    path = working_dir / filename
    if not path.is_file():
        logger.warning("%s does not exist — skipping namespace %r", path, namespace)
        return {"namespace": namespace, "skipped": True}

    data, matrix = _load_vdb_json(path)
    collection = _qdrant_collection_name(namespace, collection_prefix)
    logger.info(
        "[%s] loaded %d records (dim=%d) from %s -> collection %r (workspace=%r)",
        namespace, len(data), matrix.shape[1] if matrix.size else EXPECTED_EMBEDDING_DIM,
        path, collection, workspace,
    )

    points: list[Any] = []
    for i, rec in enumerate(data):
        record_id = rec["__id__"]
        payload: dict[str, Any] = {
            ID_FIELD: record_id,
            WORKSPACE_ID_FIELD: workspace,
            CREATED_AT_FIELD: rec.get("__created_at__"),
        }
        for key in meta_fields:
            if key in rec:
                payload[key] = rec[key]
        points.append(
            models.PointStruct(
                id=_compute_point_id(record_id, workspace),
                vector=matrix[i].tolist(),
                payload=payload,
            )
        )

    upserted = 0
    t0 = time.monotonic()
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch, wait=True)
        upserted += len(batch)
        if upserted % (batch_size * 10) == 0 or upserted == len(points):
            logger.info("[%s] upserted %d/%d", namespace, upserted, len(points))
    elapsed = time.monotonic() - t0

    actual_count = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key=WORKSPACE_ID_FIELD, match=models.MatchValue(value=workspace))]
        ),
        exact=True,
    ).count
    if actual_count != len(data):
        raise RuntimeError(
            f"[{namespace}] post-upsert count mismatch: expected {len(data)}, "
            f"Qdrant reports {actual_count} for workspace {workspace!r}. Do not proceed."
        )
    logger.info(
        "[%s] OK — %d points confirmed in %r (workspace=%r) in %.1fs",
        namespace, actual_count, collection, workspace, elapsed,
    )
    return {
        "namespace": namespace,
        "collection": collection,
        "source_records": len(data),
        "qdrant_count": actual_count,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Reverse: Qdrant -> JSON (rollback / export)
# ---------------------------------------------------------------------------


def _vector_to_nanovdb_field(vector: list[float]) -> str:
    """Reproduce NanoVectorDBStorage.upsert()'s per-record 'vector' encoding
    exactly: float32 -> float16 -> zlib -> base64 (lightrag/kg/nano_vector_db_impl.py)."""
    vector_f16 = np.asarray(vector, dtype=np.float32).astype(np.float16)
    compressed = zlib.compress(vector_f16.tobytes())
    return base64.b64encode(compressed).decode("utf-8")


def transplant_namespace_reverse(
    *,
    client: Any,
    working_dir: Path,
    namespace: str,
    workspace: str,
    collection_prefix: str,
    scroll_batch_size: int,
) -> dict[str, Any]:
    from qdrant_client import models

    filename, meta_fields = NAMESPACES[namespace]
    collection = _qdrant_collection_name(namespace, collection_prefix)
    if not client.collection_exists(collection):
        logger.warning("Collection %r does not exist — skipping namespace %r", collection, namespace)
        return {"namespace": namespace, "skipped": True}

    data: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    offset = None
    workspace_filter = models.Filter(
        must=[models.FieldCondition(key=WORKSPACE_ID_FIELD, match=models.MatchValue(value=workspace))]
    )
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=workspace_filter,
            limit=scroll_batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            payload = dict(point.payload or {})
            record: dict[str, Any] = {
                "__id__": payload.get(ID_FIELD),
                "__created_at__": payload.get(CREATED_AT_FIELD),
            }
            for key in meta_fields:
                if key in payload:
                    record[key] = payload[key]
            vector = point.vector
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            record["vector"] = _vector_to_nanovdb_field(vector)
            data.append(record)
            vectors.append(vector)
        if next_offset is None:
            break
        offset = next_offset

    if not data:
        logger.warning("[%s] no points found for workspace %r in %r", namespace, workspace, collection)
        matrix = np.zeros((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
    else:
        matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, EXPECTED_EMBEDDING_DIM)

    out_doc = {
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "data": data,
        "matrix": base64.b64encode(matrix.tobytes()).decode("utf-8"),
    }
    working_dir.mkdir(parents=True, exist_ok=True)
    out_path = working_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out_doc, fh, ensure_ascii=False)
    logger.info("[%s] exported %d records -> %s", namespace, len(data), out_path)
    return {"namespace": namespace, "collection": collection, "exported_records": len(data), "path": str(out_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--working-dir",
        default=os.environ.get("HARS_MEMORY_INDEX_DIR"),
        required="HARS_MEMORY_INDEX_DIR" not in os.environ,
        help=(
            "Forward: source of vdb_*.json. Reverse: destination for rebuilt vdb_*.json. "
            "No machine-specific default (env: HARS_MEMORY_INDEX_DIR) — matches "
            "--collection-prefix's required-config pattern below."
        ),
    )
    ap.add_argument("--qdrant-url", default=os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335"))
    ap.add_argument(
        "--workspace",
        default=os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory"),
        help="Tenant id written to/read from workspace_id — NOT a Qdrant collection name.",
    )
    ap.add_argument(
        "--collection-prefix",
        default=os.environ.get("HARS_MEMORY_QDRANT_COLLECTION_PREFIX"),
        required="HARS_MEMORY_QDRANT_COLLECTION_PREFIX" not in os.environ,
        help=(
            "Per-project prefix applied to the fixed lightrag_vdb_{chunks,entities,"
            "relationships} collection names, so multiple projects can share one "
            "Qdrant container without colliding on schema. Required (env: "
            "HARS_MEMORY_QDRANT_COLLECTION_PREFIX) — no default."
        ),
    )
    ap.add_argument(
        "--namespace",
        choices=[*NAMESPACES.keys(), "all"],
        default="all",
        help="Restrict to a single namespace (default: all three).",
    )
    ap.add_argument("--batch-size", type=int, default=500, help="Forward upsert / reverse scroll batch size.")
    ap.add_argument("--reverse", action="store_true", help="Qdrant -> vdb_*.json instead of the default JSON -> Qdrant.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.qdrant_url, timeout=30)
    working_dir = Path(args.working_dir)
    namespaces = list(NAMESPACES.keys()) if args.namespace == "all" else [args.namespace]

    results = []
    for namespace in namespaces:
        if args.reverse:
            result = transplant_namespace_reverse(
                client=client,
                working_dir=working_dir,
                namespace=namespace,
                workspace=args.workspace,
                collection_prefix=args.collection_prefix,
                scroll_batch_size=args.batch_size,
            )
        else:
            result = transplant_namespace_forward(
                client=client,
                working_dir=working_dir,
                namespace=namespace,
                workspace=args.workspace,
                collection_prefix=args.collection_prefix,
                batch_size=args.batch_size,
            )
        results.append(result)

    print(json.dumps({"mode": "reverse" if args.reverse else "forward", "results": results}, indent=2))
    if any(r.get("skipped") for r in results):
        sys.exit(0)


if __name__ == "__main__":
    main()
