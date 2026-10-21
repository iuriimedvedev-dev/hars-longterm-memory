# hars-longterm-memory on Kubernetes

Kustomize bundle for the native HTTP(S) MCP endpoint
(`hars-longterm-memory-mcp-https`, `hars_memory.http_server:main`) with the
built-in OAuth 2.1 authorization server.

Topology, in one sentence: **one** engine pod behind a ClusterIP Service,
TLS-terminated at ingress-nginx in the work profile, vectors in an **external**
Qdrant, the LightRAG working directory and OAuth state on a **ReadWriteOnce**
PVC.

```
client ──https──> ingress-nginx ──http──> Service:8099 ──> Pod (1 replica)
                                                             ├─ /data (PVC): index, staging, OAuth state
                                                             ├─ Qdrant (external, lightrag_vdb_*)
                                                             └─ embedding / LLM endpoints (external)
```

## Files

| Path | Purpose |
|---|---|
| `base/` | shared resources: Namespace, ConfigMap, PVC, Deployment, Service. **OAuth off, no Ingress.** |
| `components/ingress/` | Ingress for `/mcp` + all OAuth endpoints (work only) |
| `components/oauth-signing-key/` | mounts the signing key from a Secret (work only) |
| `components/app-tls/` | optional app-terminated TLS (topology W2) |
| `components/image-pull-secret/` | work profile `imagePullSecrets: registry-credentials` |
| `overlays/local/` | home profile: no OAuth, no Ingress, `local-path`, local registry |
| `overlays/work/` | enterprise profile: OAuth, Ingress, TLS (A/B), `fast-ssd`, private registry |
| `jobs/restore-index/` | optional first-run index snapshot restore |
| `jobs/qdrant-transplant/` | optional NanoVectorDB -> Qdrant migration |
| `qdrant-statefulset.yaml` | optional single-replica Qdrant (external by default) |
| `secret.example.yaml` | placeholders only (signing key, approval token, Qdrant, restore, TLS) |
| `Dockerfile` | minimal image for the HTTP server (no index baked in) |
| `kustomization.yaml` | default = `overlays/work` |

## Prerequisites

* Kubernetes >= 1.25 (Pod Security Admission `restricted` is enforced).
* `kubectl` >= 1.25 (bundles a kustomize new enough for `components`/`patches`).
* An ingress controller for the work profile (annotations target
  **ingress-nginx**). The local profile needs none.
* A StorageClass: `local-path` for home, `fast-ssd` placeholder for work.
* An external Qdrant, **or** apply the optional `qdrant-statefulset.yaml`.
* An OpenAI-compatible embedding endpoint whose model produces the **same
  vector dimension as the index** (768 for the embeddinggemma-300m /
  e5-large-v2 generation).
* For `memory_remember` / `memory_upsert_document` only: an extraction LLM.
  Recall works without it.
* TLS: cert-manager **or** a pre-created `kubernetes.io/tls` secret (work).

## Image build and push

```bash
# from the repo root; the Dockerfile lives in this bundle
docker build -f deploy/k8s/Dockerfile -t hars-longterm-memory:0.8.1 .
docker tag  hars-longterm-memory:0.8.1 registry.local/hars-longterm-memory:0.8.1
docker push registry.local/hars-longterm-memory:0.8.1

# work registry
docker tag  hars-longterm-memory:0.8.1 registry.work.example/hars/hars-longterm-memory:0.8.1
docker push registry.work.example/hars/hars-longterm-memory:0.8.1
```

The tag is consumed by the `images:` entry of each overlay — that is knob 4.
No index is baked into the image; it comes from the PVC and Qdrant.

## Where Qdrant sits

The default is an **external** Qdrant (`http://qdrant.qdrant.svc.cluster.local:6333`,
i.e. namespace `qdrant`), not a sidecar:

* one Qdrant can serve several memory projects with distinct
  `HARS_MEMORY_QDRANT_COLLECTION` values;
* it survives engine rollbacks and keeps vectors out of the 10 GiB index PVC;
* the engine only needs network access to it, no shared filesystem.

`qdrant-statefulset.yaml` is optional and exists only for clusters that have no
Qdrant. It creates the `qdrant` namespace and the exact Service name the
ConfigMap expects. Apply either it **or** point the ConfigMap at an existing
endpoint — never both, and do not also apply the code-diver bundle's embedded
Qdrant in the same cluster.

## Profiles
| | `overlays/local` | `overlays/work` |
|---|---|---|
| OAuth | **off** (`HARS_MEMORY_MCP_OAUTH_ENABLED=0`) | **on**, issuer = public host |
| Exposure | ClusterIP only, plain HTTP, no Ingress | ingress-nginx + TLS |
| TLS | none (app plain HTTP) | ingress terminates TLS (W1, default) or app TLS + passthrough (W2) |
| StorageClass | `local-path` | `fast-ssd` (placeholder) |
| Registry | `registry.local/hars-longterm-memory:0.8.1` | `registry.work.example/...` + pull secret |
| Secrets | **none required** | `hars-memory-oauth`, plus TLS (unless cert-manager) |
| Resources | 1 CPU / 2 Gi request | 2 CPU / 4 Gi request |

## The four knobs (never edit `base/`)

| Knob | Where | Local value | Work value |
|---|---|---|---|
| 1. Issuer / public host | `overlays/work/configmap.patch.yaml` + `overlays/work/kustomization.yaml` (JSON6902) | n/a (OAuth off) | `memory.work.example` |
| 2. TLS | `overlays/work/kustomization.yaml` -> variant A `ingress-cert-manager.patch.yaml` (active) or B `ingress-external-secret.patch.yaml` | no Ingress | A or B |
| 3. StorageClass | `overlays/*/pvc.patch.yaml` | `local-path` | `fast-ssd` |
| 4. Registry | `images:` in `overlays/*/kustomization.yaml` | `registry.local/...` | `registry.work.example/...` |

## Human decisions (the four knobs, plus dependencies)

| Decision | Default in this bundle | Where |
|---|---|---|
| **Issuer / public host** | work: `memory.work.example`; local: none (OAuth off) | work `configmap.patch.yaml` + `kustomization.yaml` |
| **TLS: cert-manager vs BYO secret** | work variant A (cert-manager `letsencrypt-prod`), variant B present but inactive | work `kustomization.yaml` + `ingress-*.patch.yaml` |
| **StorageClass** | local `local-path`, work `fast-ssd` | `overlays/*/pvc.patch.yaml` |
| **Image registry** | `registry.local/...`, `registry.work.example/...` (+ `registry-credentials` pull secret) | `overlays/*/kustomization.yaml`, `components/image-pull-secret` |
| **Qdrant endpoint** | external `qdrant.qdrant.svc.cluster.local:6333` or optional StatefulSet | `base/configmap.yaml` (`HARS_MEMORY_QDRANT_URL`) |
| **Embedder + LLM endpoints** | `embedding.embedding.svc...`, `llm.llm.svc...` | `base/configmap.yaml` |
| **TLS topology** | W1 ingress termination (W2 app termination available) | `components/app-tls` + work overlay |

Everything else (probes, resources, securityContext, `Recreate`, guards) is in
`base/` and should not change per cluster.

## Deploy locally in 5 commands

```bash
# 1. (optional) point at a local registry and storage class
sed -i 's#registry.local/hars-longterm-memory#registry.home.lan/hars-longterm-memory#' \
  deploy/k8s/overlays/local/kustomization.yaml
sed -i 's/local-path/my-storage-class/' deploy/k8s/overlays/local/pvc.patch.yaml
# kind/minikube with no registry: use newName: hars-longterm-memory (no host)
# and `kind load docker-image hars-longterm-memory:0.8.1`; base is IfNotPresent.

# 2. make sure Qdrant is reachable (external) or apply the optional one once
kubectl apply -f deploy/k8s/qdrant-statefulset.yaml   # skip if you have Qdrant

# 3. apply the local profile (no secrets, no Ingress, OAuth off)
kubectl apply -k deploy/k8s/overlays/local

# 4. wait for the pod
kubectl -n hars-memory rollout status deploy/hars-memory-mcp --timeout=5m

# 5. expose it on localhost (keep this on a trusted machine / tailnet)
kubectl -n hars-memory port-forward svc/hars-memory-mcp 8099:8099
```

Verify (in another shell):

```bash
curl -s http://127.0.0.1:8099/healthz | jq
# status ok, index_warm true, oauth false
python scripts/mcp_http_smoke.py --url http://127.0.0.1:8099 --no-auth
```

Client config: `{"mcpServers":{"hars-longterm-memory":{"type":"http","url":"http://127.0.0.1:8099/mcp"}}}`.

The local profile is **unauthenticated**. Never expose it beyond the cluster
network, a WireGuard/tailnet interface, or the port-forward above. To publish
it on a LAN, enable the optional NodePort patch in
`overlays/local/kustomization.yaml` — only on a single-user cluster.

## Deploy at work in 5 commands

```bash
# 1. set the four knobs (issuer host appears in two files; registry/tls/storage)
sed -i 's/memory\.work\.example/memory.acme.internal/g' \
  deploy/k8s/overlays/work/configmap.patch.yaml deploy/k8s/overlays/work/kustomization.yaml
sed -i 's#registry\.work\.example/hars/hars-longterm-memory#registry.acme.internal/hars/hars-longterm-memory#' \
  deploy/k8s/overlays/work/kustomization.yaml
sed -i 's/fast-ssd/gp3/' deploy/k8s/overlays/work/pvc.patch.yaml
sed -i 's/letsencrypt-prod/acme-prod/' deploy/k8s/overlays/work/ingress-cert-manager.patch.yaml

# 2. namespace + OAuth secret (Ed25519 signing key + approval token)
kubectl create namespace hars-memory
openssl genpkey -algorithm ed25519 -out /tmp/signing.key
kubectl -n hars-memory create secret generic hars-memory-oauth \
  --from-file=signing-key.pem=/tmp/signing.key \
  --from-literal=approval-token="$(openssl rand -hex 24)" \
  --from-literal=authorize-password="$(openssl rand -hex 16)"
rm -f /tmp/signing.key

# 2b. variant B only: bring your own certificate instead of cert-manager
#     (edit kustomization.yaml first: comment A, uncomment B -- see "TLS")
# kubectl -n hars-memory create secret tls hars-memory-tls --cert=fullchain.pem --key=privkey.pem

# 3. (optional) pull secret for the private registry
kubectl -n hars-memory create secret docker-registry registry-credentials \
  --docker-server=registry.acme.internal \
  --docker-username='...' --docker-password='...'

# 4. apply
kubectl apply -k deploy/k8s/overlays/work

# 5. wait, then verify discovery through the real host
kubectl -n hars-memory rollout status deploy/hars-memory-mcp --timeout=5m
curl -s https://memory.acme.internal/.well-known/oauth-protected-resource | jq
curl -s https://memory.acme.internal/.well-known/oauth-authorization-server | jq
```

The OAuth issuer in the ConfigMap and the Ingress host **must be identical**;
if they differ, clients reject the metadata. Both are patched from the same
`sed` in step 1.

## Secrets

| Secret | Keys | Required by |
|---|---|---|
| `hars-memory-oauth` | `signing-key.pem`, `approval-token`, `authorize-password` | work profile (OAuth) |
| `hars-memory-qdrant` | `api-key` | only if Qdrant needs auth |
| `hars-memory-tls` | `tls.crt`, `tls.key` | TLS variant B |
| `hars-memory-mcp-tls` | `tls.crt`, `tls.key` | topology W2 (app TLS) |
| `hars-memory-restore` | `snapshot-url`, `snapshot-sha256` | restore Job |
| `registry-credentials` | dockerconfigjson | private registry |

`secret.example.yaml` contains **placeholders only**. Create the real objects
with `kubectl create secret ...` as above; never commit or `kubectl apply` the
example file.

OAuth fails closed: with `HARS_MEMORY_MCP_OAUTH_ENABLED=1` and neither
`approval-token` nor `authorize-password`, the server refuses to start.
Rotating the signing key invalidates every issued token. Deleting
`/data/oauth/state.json` (while stopped) revokes every client.

## TLS topologies

**W1 (default, active):** ingress-nginx terminates TLS, the pod speaks HTTP.
* Variant A: `ingress-cert-manager.patch.yaml` adds
  `cert-manager.io/cluster-issuer: letsencrypt-prod`; cert-manager creates
  `hars-memory-tls`.
* Variant B: comment A, uncomment `ingress-external-secret.patch.yaml`, create
  `hars-memory-tls` yourself; the `null` annotation prevents cert-manager from
  touching it.

**W2 (optional):** the app terminates TLS; the ingress passes the stream
through.
1. uncomment `- ../../components/app-tls` in `overlays/work/kustomization.yaml`
2. uncomment `- path: ingress-passthrough.patch.yaml`
3. create `hars-memory-mcp-tls` (same cert/key)
4. ingress-nginx must run with `--enable-ssl-passthrough`
The component also flips the three probes to `scheme: HTTPS`.

**Local:** no TLS anywhere. If you need HTTPS on a home network, use a
self-signed certificate with variant W2 and the same `app-tls` component.

`X-Forwarded-Proto`: ingress-nginx sets it; the app trusts it only from
`FORWARDED_ALLOW_IPS` (`base/configmap.yaml`, default `"*"` because the pod is
ClusterIP-only). Narrow it to the ingress controller's pod CIDR when the
cluster network is not fully trusted.

## Apply order, restore, migration

Fresh cluster, empty index:

```bash
kubectl apply -k deploy/k8s/overlays/work   # or overlays/local
kubectl -n hars-memory rollout status deploy/hars-memory-mcp --timeout=5m
```

Existing index snapshot, restore **before** the engine serves traffic. The
overlay creates the namespace and the PVC with its final StorageClass; the
engine must be stopped while the snapshot is unpacked (the Job clears
`/data/index` first, `CLEAN_TARGET=1`):

```bash
kubectl apply -k deploy/k8s/overlays/work          # or overlays/local
kubectl -n hars-memory scale deploy/hars-memory-mcp --replicas=0

kubectl -n hars-memory create secret generic hars-memory-restore \
  --from-literal=snapshot-url='https://artifacts.example.com/hars/index-snapshot.tar.gz' \
  --from-literal=snapshot-sha256='<sha256>'
kubectl -n hars-memory apply -k deploy/k8s/jobs/restore-index
kubectl -n hars-memory wait --for=condition=complete job/hars-memory-restore-index --timeout=1h

kubectl -n hars-memory scale deploy/hars-memory-mcp --replicas=1
```

The tarball must contain the **contents** of the LightRAG working directory at
its root (graphml, `kv_store_*.json`, `vdb_*.json`/Qdrant workspace config,
BM25 sidecars), not a parent folder. The Job is idempotent
(`/data/index/.restore-complete`); set `FORCE_RESTORE=1` to overwrite again.

Migration from a file-based index to Qdrant (one-off, engine stopped):

```bash
kubectl -n hars-memory scale deploy/hars-memory-mcp --replicas=0
kubectl -n hars-memory apply -k deploy/k8s/jobs/qdrant-transplant
kubectl -n hars-memory logs -f job/hars-memory-qdrant-transplant
kubectl -n hars-memory scale deploy/hars-memory-mcp --replicas=1
```

The Job copies the three `vdb_*.json` stores into
`lightrag_vdb_{chunks,entities,relationships}` with **no re-embedding**. Keep a
copy of the index directory, or use the reverse mode in
`hars_memory/scripts/qdrant_transplant.py --reverse`, before relying on it.

## Verify

```bash
kubectl -n hars-memory get pod -l app.kubernetes.io/name=hars-longterm-memory
kubectl -n hars-memory logs deploy/hars-memory-mcp --tail=50

# liveness + warm state (public)
kubectl -n hars-memory port-forward svc/hars-memory-mcp 8099:8099 &
curl -s http://127.0.0.1:8099/healthz | jq
# expect: status ok, index_warm true, warmup_seconds ~17-21, oauth true

# discovery through the ingress
curl -s https://memory.acme.internal/.well-known/oauth-protected-resource | jq
curl -s https://memory.acme.internal/.well-known/oauth-authorization-server | jq

# end-to-end: DCR -> PKCE -> token -> initialize -> tools/list -> memory_recall
python scripts/mcp_http_smoke.py --url https://memory.acme.internal \
  --approval-token "$HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN" \
  --clients 1 --repeat 1
```

## Scaling and the engine/edge split

**Do not set `replicas: 2`.** This is a single-writer engine:

1. the LightRAG working directory (GraphML + KV + doc status) is opened and
   mutated by one process, on a RWO volume;
2. OAuth state (one JSON file: clients/refresh/revocations) is also single-writer;
3. a **bounded in-process** thread pool (`HARS_MEMORY_MCP_ENGINE_THREADS`,
   `_QUEUE`, `_QUEUE_TIMEOUT_MS`) exists in the `feat/mcp-scaling` working tree
   but was not committed at the time of writing; the bundle ships those vars
   commented out. It bounds retrieval threads per process; it does **not**
   implement an engine/edge split, and it does not change the writer count.
   There is no separate engine and edge workload in this bundle because the
   code for it is not on the branch.

Once merged, the thread pool keeps concurrent clients from serializing on the
event loop. The embedder and Qdrant become the ceiling before the pool does.

If you need more throughput, scale **vertically** first (CPU/RAM), then split:

* shared Qdrant (already the default) makes vector reads multi-process safe;
* the index directory must move to a shared store or become read-only for
  readers — LightRAG has no cross-process cache invalidation;
* OAuth state must move to a shared store (the SQLite/Redis store is on the
  scaling branch but not in the committed server yet; enable
  `HARS_MEMORY_MCP_OAUTH_STORE` once merged), and the signing key must be
  identical across pods;
* exactly one process may still write the index. A future edge/engine split
  would put stateless HTTP edges in front of one engine, not two engines.

Until that exists, one replica is the correct topology.

## Switching profiles

Applying one overlay over the other leaves objects the new profile no longer
declares (typically the Ingress). Switch explicitly:

```bash
# work -> local: remove the Ingress (and optionally the pull secret)
kubectl -n hars-memory delete ingress hars-memory-mcp --ignore-not-found
kubectl apply -k deploy/k8s/overlays/local

# local -> work
kubectl apply -k deploy/k8s/overlays/work
```

Secrets and PVCs are never deleted by a profile switch. The ConfigMap is
updated in place, so a rollout follows (`kubectl rollout restart`).

## Rollback

```bash
kubectl -n hars-memory rollout undo deploy/hars-memory-mcp
kubectl -n hars-memory rollout status deploy/hars-memory-mcp
```

The PVC has no owner reference: deleting the Deployment or namespace does not
delete the index. Back it up as a `tar.gz` of `/data/index` plus a Qdrant
snapshot; that pair is what the restore/transplant jobs consume. Rolling back a
bad *index* means the snapshot restore above, with the engine scaled to 0.

## Validation status

Validated on the build host:

* every YAML file parses with `yaml.safe_load_all`;
* required fields (`apiVersion`/`kind`/`metadata.name`), probes, resources,
  securityContext and PVC references were asserted by script;
* kustomization resource/component paths and patch targets were resolved and
  checked (including prefix/suffix and namespace rewrites);
* the bundle was grepped for real secrets, private IPs and host-specific paths.

**Not** validated: `kubectl apply --dry-run`, `kubectl kustomize`, schema
validation against a live API server, and runtime behaviour. Neither `kubectl`
nor `kustomize` is installed on the machine that produced these files, so no
cluster-side dry run was performed. Run
`kubectl apply -k deploy/k8s/overlays/work --dry-run=server` before the first
real apply.
