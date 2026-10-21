# Kubernetes deployment (HTTP MCP + OAuth)

The manifests live in [`deploy/k8s/`](../../deploy/k8s/README.md). That README
is the operational reference (apply order, secrets, restore jobs, rollback);
this page records the deployment topology and the **embedder sizing** rules.

## Topology

```
client ──https──> ingress-nginx ──http──> Service:8099 ──> Pod (1 replica)
                                                             ├─ /data (PVC): index, staging, OAuth state
                                                             ├─ Qdrant (external, lightrag_vdb_*)
                                                             └─ embedder (external URL, or optional in-cluster llama-server)
```

* One `hars-longterm-memory-mcp-https` process, `replicas: 1`,
  `strategy: Recreate`, ReadWriteOnce index PVC.
* TLS terminates at ingress-nginx (work profile) or is absent (local profile);
  app-terminated TLS is an optional component (W2).
* OAuth is **off** in the local profile and **on** in the work profile, with a
  per-cluster issuer host.
* Vectors live in Qdrant so they survive engine rollbacks; the index working
  directory holds the graph/KV and is not multi-writer.

## Embedder is the first bottleneck

Measured on the reference CPU-only host (16 cores) with llama-server at
`-t 4 -np 2`:

| Load | Throughput vs 1 client |
|---|---|
| 8 concurrent embedding requests | **1.46×** |
| 8 concurrent `memory_recall` calls | **1.51×** |

The engine processes used only ~0.3–0.6 cores while throughput saturated: the
requests were queued on two embedding slots. Adding app replicas against the
same endpoint lengthens the queue; it does not raise throughput.

### Concurrency model

* One uvicorn worker and one shared index per pod.
* With the bounded engine pool merged (`HARS_MEMORY_MCP_ENGINE_THREADS`,
  default `min(8, cpus)`), up to that many retrieval calls run concurrently;
  the queue is `4 × threads`. The pool is not in the committed server yet, so
  the bundle ships those variables commented out; without it, blocking
  retrieval runs inline on the event loop.
* One recall issues ~one embedding request (`flat_dense` is off by default), so
  **slots ≈ simultaneous recalls**. Write tools also embed and compete.
* Parallel: HTTP handling, embedder slots, Qdrant, numpy. Serial: Python
  fusion/assembly per process (GIL), index writes, OAuth state writes.

### Sizing table

| Simultaneous recalls | `-np` slots | `-t` threads | Pod CPU req/limit | Memory req/limit |
|---|---|---|---|---|
| 1–2 (home) | 2 | 4 | 1 / 2 | 1Gi / 2Gi |
| 3–4 | 4 | 4 | 2 / 4 | 1Gi / 2Gi |
| 5–8 (work default) | 8 | 6–8 | 4 / 6 | 2Gi / 4Gi |
| 9–16 | 12–16 | 8 | 6 / 8 | 2Gi / 4Gi |
| >16 | second embedder Service + split groups, or GPU | | | |

`EMBED_THREADS` must not exceed the CPU limit; a lower limit throttles every
slot. Keep `EMBED_CTX=2048`, `EMBED_BATCH/UBATCH=512` for 300M models.

### Enabling the in-cluster embedder

```bash
# uncomment in deploy/k8s/overlays/{work,local}/kustomization.yaml:
#   - ../../components/embedder
#   - path: embedder-sizing.patch.yaml
#   - path: embedder-resources.patch.yaml
#   - path: embedder-storageclass.patch.yaml
# model: create Secret embedder-model/url (init container downloads it) or
# pre-seed the embedder-models volume with the GGUF before the pod starts
kubectl apply -k deploy/k8s/overlays/work
```

The component creates a single-replica llama-server (`/app/llama-server`,
OpenAI-compatible `/v1/embeddings` on 8791) with a 5 GiB model PVC, an
init container that downloads the GGUF when a URL is configured, probes on
`/health`, and hardening (non-root, read-only rootfs, drop ALL). It rewrites
`HARS_MEMORY_EMBED_BASE_URL` to `http://embedder:8791/v1`.

### Measuring

```bash
# embedder: compare 1 vs N concurrent /v1/embeddings requests (README script)
kubectl -n hars-memory port-forward svc/embedder 8791:8791
# service: 8 clients, 5 recalls each
python scripts/mcp_http_smoke.py --url https://<issuer> \
  --approval-token "$HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN" --clients 8 --repeat 5
# engine stats (when the pool is enabled): engine threads/queue, loop lag
curl -s https://<issuer>/healthz | jq '{index_warm, engine, loop}'
```

### Rule

1. Scale the **embedder** (slots, then threads/CPU) before adding app replicas.
2. `replicas > 1` needs shared Qdrant (default), a shared OAuth store, an
   identical signing key, and exactly one index writer. The committed server
   does not have the shared OAuth store yet, so keep one replica per engine.
3. For more projects, add namespaces/engines rather than replicas. A future
   edge/engine split is not implemented.

## Validation status

The bundle was validated by YAML parsing plus a reduced kustomize renderer
(resources/components/patches/prefixes, probes, security, PVC references,
profile invariants, embedder composition). `kubectl`/`kustomize` are not
installed on the build host, so no `--dry-run` was performed. Run
`kubectl apply -k deploy/k8s/overlays/work --dry-run=server` before the first
apply.
