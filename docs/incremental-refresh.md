# Incremental refresh (`hars_memory.scripts.incremental_refresh`)

`plan | apply | audit | repair-orphans` for a LightRAG index (JSON KV stores + graphml + Qdrant). Same environment as `memory-index`.

- `plan` (read-only, no LLM, no embedder): classifies every walked document against the fingerprint sidecar AND the content LightRAG actually stored (`kv_store_full_docs.json`, after LightRAG's sanitising): `new`, `changed`, `retry` (doc_status exists but is not `processed`), `adopt` (processed, no fingerprint, content matches: record only), `unchanged`, `gone` (fingerprinted, no longer walked), `kept` (protected by `--keep-ids-file`). GC is refused when the walk covers < 50 % of the fingerprinted ids.
- `apply`: an unchanged index is a strict no-op (does not even start LightRAG). Otherwise one batched deletion (LightRAG's own `adelete_by_doc_id` per document, but the entity/relation rebuild and the storage flushes are deferred and run once for the union), then inserts in verified batches; fingerprints are persisted after each verified batch.
- `audit`: KV stores vs graph vs Qdrant (`--prefixes "" pier_kb` also compares the `pier_kb_lightrag_vdb_*` copies). `repair-orphans` removes Qdrant points that have no KV/graph counterpart (left behind by a killed run).
- `HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL` (`ingest/document.py`): doc ids are `sha256(absolute path)`; the map lets a refresh run from another checkout location reproduce the ids stored in the index.

Why not `memory-index --refresh-changed`: it passes ALL documents to `ainsert` (one `dup-...` doc_status record per document per run), deletes with one full rebuild + two full flushes per document (60-180 s each on the Pier index), and saves fingerprints only at the end of the run.
