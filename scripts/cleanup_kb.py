#!/usr/bin/env python3
"""Knowledge-base cleanup: purge stale documents from the GraphRAG index by
document date, with keyword protection for knowledge that must survive.

Every corpus/staging doc carries a `[Document: ... | Date: YYYY-MM-DD]` header;
that date (not ingest time) is the relevance date. Deleting a doc removes its
chunks and prunes entities/relations that lose all their sources (LightRAG
``adelete_by_doc_id``).

Usage (dry-run by default — prints what WOULD be deleted):
    uv run --project tools/graphrag python tools/graphrag/scripts/cleanup_kb.py \
        --before 2026-05-01 \
        --keep 'falsif|rocm|therock|gfx1201|golden|methodology' \
        --sections session,db \
        [--apply]

    --before DATE | --older-than-days N   cutoff (doc Date < cutoff = candidate)
    --keep REGEX     protect docs whose filename OR content matches (repeatable)
    --sections CSV   only consider these sections (default: all)
    --apply          actually delete (otherwise dry-run)
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HEADER_RE = re.compile(
    r"\[Document:\s*(?P<name>[^|\]]+)\|\s*Section:\s*(?P<section>[^|\]]+)"
    r"\|\s*Date:\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}|unknown)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    cut = ap.add_mutually_exclusive_group(required=True)
    cut.add_argument("--before", type=str, help="delete docs dated before YYYY-MM-DD")
    cut.add_argument("--older-than-days", type=int, help="delete docs older than N days")
    ap.add_argument("--keep", action="append", default=[],
                    help="protection regex (filename or content match) — repeatable")
    ap.add_argument("--sections", type=str, default="",
                    help="comma-separated sections to consider (default all)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default dry-run)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.before:
        cutoff = dt.date.fromisoformat(args.before)
    else:
        cutoff = dt.date.today() - dt.timedelta(days=args.older_than_days)
    keep_res = [re.compile(p, re.IGNORECASE) for p in args.keep]
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}

    wdir = Path(os.environ.get("GRAPHRAG_WORKING_DIR", "/mnt/datasets/graphrag/index_gemma_v4"))
    full_docs = json.load(open(wdir / "kv_store_full_docs.json"))
    status = json.load(open(wdir / "kv_store_doc_status.json"))

    victims: list[tuple[str, str, str, str]] = []   # (doc_id, fname, section, date)
    kept_by_filter = 0
    undated = 0
    for doc_id, rec in full_docs.items():
        content = rec.get("content", "") if isinstance(rec, dict) else str(rec)
        m = HEADER_RE.search(content[:400])
        fname = (status.get(doc_id, {}) or {}).get("file_path") or (m.group("name").strip() if m else doc_id)
        section = m.group("section").strip() if m else "?"
        date_s = m.group("date") if m else "unknown"
        if sections and section not in sections:
            continue
        if date_s == "unknown":
            undated += 1
            continue                      # never delete undated docs
        if dt.date.fromisoformat(date_s) >= cutoff:
            continue
        if any(r.search(fname) or r.search(content) for r in keep_res):
            kept_by_filter += 1
            continue
        victims.append((doc_id, fname, section, date_s))

    print(f"index: {wdir} | docs total: {len(full_docs)} | cutoff: < {cutoff}")
    print(f"candidates: {len(victims)} | protected by --keep: {kept_by_filter} | undated (never touched): {undated}")
    by_sec: dict[str, int] = {}
    for _, _, sec, _ in victims:
        by_sec[sec] = by_sec.get(sec, 0) + 1
    print("by section:", by_sec)
    for _, fname, sec, d in sorted(victims, key=lambda v: v[3])[:15]:
        print(f"  {d}  [{sec}]  {fname}")
    if len(victims) > 15:
        print(f"  ... and {len(victims) - 15} more")

    if not args.apply:
        print("\nDRY-RUN — nothing deleted. Re-run with --apply to purge.")
        return
    if not victims:
        print("nothing to delete")
        return

    os.environ.setdefault("GRAPHRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    os.environ.setdefault("GRAPHRAG_EMBED_MODEL", "unsloth/embeddinggemma-300m")
    os.environ.setdefault("GRAPHRAG_EMBED_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("GRAPHRAG_EMBED_DEVICE", "cpu")
    os.environ.setdefault("HF_HOME", "/mnt/datasets/models/.hf_home")
    from tools.graphrag.server.lightrag_init import create_lightrag

    async def purge() -> None:
        rag = create_lightrag(working_dir=str(wdir))
        await rag.initialize_storages()
        try:
            for i, (doc_id, fname, _, _) in enumerate(victims, 1):
                await rag.adelete_by_doc_id(doc_id)
                if i % 20 == 0 or i == len(victims):
                    print(f"deleted {i}/{len(victims)} (last: {fname})")
        finally:
            await rag.finalize_storages()

    asyncio.run(purge())
    print("PURGE COMPLETE")


if __name__ == "__main__":
    main()
