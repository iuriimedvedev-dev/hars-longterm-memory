#!/usr/bin/env python3
"""Generate CLEARLY SYNTHETIC demo fixtures for the Memory Travel demo mode.

No real data: generic demo entities on a torus/grid layout, a synthetic trace.
Everything is deterministic (seeded) so the demo is reproducible.
Output: app/public/data/{graph.json, coords_entities.json, coords_chunks.json,
node_index.json, traces/{manifest.json,trace_01.json}}.
"""
import json
import math
import os
import random

random.seed(20261018)
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "public", "data")

FAMILIES = ["document", "concept", "tool", "model", "dataset", "metric"]
TITLES = [
    "Demo note", "Demo spec", "Demo run", "Demo plan", "Demo report",
    "Demo architecture", "Demo config", "Demo dataset", "Demo metric", "Demo checklist",
]
WORDS = ["alpha", "vector", "graph", "index", "query", "token", "cache",
         "chunk", "probe", "trace", "router", "fusion", "demo", "signal"]

N_NODES = 320
N_EDGES = 540
N_CHUNKS = 90


def blob(seed: int, size: int = 180) -> str:
    rnd = random.Random(seed)
    words = [rnd.choice(WORDS) for _ in range(max(4, size // 7))]
    sents = []
    for i in range(0, len(words), 8):
        sents.append("The demo pipeline handles the " + " ".join(words[i:i + 8]) + ".")
    return " ".join(sents)[:size]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "traces"), exist_ok=True)

    # --- graph.json: synthetic entities on a 2-torus ---
    nodes, edges = [], []
    for i in range(N_NODES):
        fam = FAMILIES[i % len(FAMILIES)]
        ang = 2 * math.pi * i / N_NODES
        name = f"demo-{fam}-{i:04d}"
        nodes.append({
            "id": name,
            "t": fam,
            "d": f"Synthetic demo entity #{i} of family {fam}. Contains no real data.",
            "deg": 3,
        })
    seen = set()
    rnd = random.Random(42)
    for i in range(N_NODES):
        for _ in range(3):
            j = rnd.randrange(N_NODES)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            edges.append([i, j])
            if len(edges) >= N_EDGES:
                break
        if len(edges) >= N_EDGES:
            break
    json.dump({"nodes": nodes, "edges": edges}, open(os.path.join(OUT, "graph.json"), "w"))

    # --- node_index.json ---
    json.dump([n["id"] for n in nodes], open(os.path.join(OUT, "node_index.json"), "w"))

    # --- coords: UMAP-like ring layout in 2D + z on a gentle wave ---
    xy, xyz, ids = [], [], []
    chunk_xy, chunk_xyz = {}, {}
    for i in range(N_NODES):
        ang = 2 * math.pi * i / N_NODES
        r = 0.55 + 0.12 * math.sin(7 * ang)
        x, y = r * math.cos(ang) + rnd.uniform(-0.05, 0.05), r * math.sin(ang) + rnd.uniform(-0.05, 0.05)
        z = 0.25 * math.sin(3 * ang)
        xy.append([round(x, 4), round(y, 4)])
        xyz.append([round(x, 4), round(y, 4), round(z, 4)])
        ids.append(nodes[i]["id"])
    for k in range(N_CHUNKS):
        ang = 2 * math.pi * k / N_CHUNKS
        r = 0.35
        x, y = r * math.cos(ang), r * math.sin(ang)
        chunk_xy[f"chunk-{k:03d}"] = [round(x, 4), round(y, 4)]
        chunk_xyz[f"chunk-{k:03d}"] = [round(x, 4), round(y, 4), 0.0]
    json.dump({"ids": ids, "xy": xy}, open(os.path.join(OUT, "coords_entities.json"), "w"))
    json.dump(chunk_xyz, open(os.path.join(OUT, "coords_chunks.json"), "w"))

    # --- traces: one synthetic trace ---
    q = "How does the demo pipeline connect the vector and graph stages?"
    chunks = []
    for k in range(24):
        chunks.append({
            "chunk_id": f"demo-chunk-{k:03d}",
            "file": f"demo-note-{k % 12:02d}.md",
            "score": round(0.95 - 0.03 * k, 4),
            "preview": blob(1000 + k, 160),
        })
    trace = {
        "meta": {"id": "demo-01", "question": q, "mode": "hybrid", "source": "synthetic-demo"},
        "query": {"text": q, "xy": [0.0, 0.0], "xyz": [0.0, 0.0, 0.0]},
        "vector_hits": chunks[:12],
        "bm25_hits": chunks[8:20],
        "fusion": chunks[:16],
        "rerank": {"applied": True, "backend": "demo", "order": list(range(12))[:10], "latency_ms": 12},
        "graph": {"nodes": [{"id": n["id"], "t": n["t"]} for n in nodes[:40]],
                  "edges": [[a, b] for a, b in edges[:60]]},
        "context_blocks": [{"ref": f"S{k+1}", "file": f"demo-note-{k%12:02d}.md",
                            "chars": len(c["preview"]) if isinstance(c.get("preview"), str) else 200}
                           for k, c in enumerate(chunks[:10])],
        "answer_preview": "Synthetic demo answer: the demo pipeline connects the demo stages.",
        "full_chunks": chunks[:16],
    }
    json.dump(trace, open(os.path.join(OUT, "traces", "trace_01.json"), "w"), ensure_ascii=False)
    manifest = [{"id": "demo-01", "question": q, "file": "traces/trace_01.json", "kb": 4}]
    json.dump(manifest, open(os.path.join(OUT, "traces", "manifest.json"), "w"))

    print("nodes", len(nodes), "edges", len(edges), "chunks", N_CHUNKS)


if __name__ == "__main__":
    main()
