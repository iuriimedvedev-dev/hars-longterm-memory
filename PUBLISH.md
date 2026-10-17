# PUBLISH — hars-longterm-memory 0.7.0

Release prepared locally while the self-hosted GitLab SSH endpoint was believed
down. **Nothing was pushed, tagged, or uploaded.** This file is the one-command
checklist for the moment GitLab is back.

- **Project:** `code.knotgate.com/hars/hars-longterm-memory` (project id `14`)
- **SSH remote (`origin`):** `ssh://git@git.knotgate.com:2222/hars/hars-longterm-memory.git`
  (note: the SSH host is `git.knotgate.com`, *not* `code.knotgate.com`)
- **PyPI endpoint:** `https://code.knotgate.com/api/v4/projects/14/packages/pypi`
- **Version:** `0.6.0` → **`0.7.0`** (minor: additive features + fixes; see CHANGELOG)
- **Artifacts built locally:** `dist/hars_longterm_memory-0.7.0-py3-none-any.whl`,
  `dist/hars_longterm_memory-0.7.0.tar.gz`

This release lives on branch **`release/0.7.0`** (worktree
`/tmp/opencode/hars-release`), branched from `main` at `991bd2f`. The live
working tree has **uncommitted in-flight work by a running agent** and was left
untouched (see "Blockers"). Rebase/merge this release branch onto the finished
`main` before tagging, then publish.

---

## 0. Re-probe GitLab availability first

```bash
# raw TCP on the two hostnames (code.knotgate.com:2222 is NOT a git endpoint)
for hp in git.knotgate.com:2222 code.knotgate.com:443; do
  h=${hp%:*}; p=${hp#*:}
  timeout 5 bash -c "echo > /dev/tcp/$h/$p" 2>/dev/null && echo "OPEN  $hp" || echo "FAIL  $hp"
done
# git over SSH (read-only auth check; prints "Welcome to GitLab, @knotgate-dev!")
ssh -T git@git.knotgate.com -p 2222
# API up? (401 = reachable, needs a token)
curl -sS -m 5 -o /dev/null -w '%{http_code}\n' https://code.knotgate.com/api/v4/version
```

## 1. Merge the release onto the finished main, then push

```bash
cd /home/angkira/Project/software/hars-longterm-memory
# (after the running agent has committed its work and `git status` is clean)
git rebase main release/0.7.0        # or: git merge release/0.7.0 into main
git checkout main && git merge --ff-only release/0.7.0
git push origin main
```

## 2. Tag and push the tag

```bash
git tag -a v0.7.0 -m "Release 0.7.0"
git push origin v0.7.0
```

The tag triggers `.gitlab-ci.yml`: `lint` → `test` → `package` →
`publish-pypi` (needs lint+test+package) → `publish-container`.
**If `lint` or `test` fails, `publish-pypi`/`publish-container` are skipped**
(see Blockers — both stages currently fail on the pre-existing issues listed
there, so fix or bypass them before relying on CI).

## 3. Manual `uv publish` (works even if CI publish is skipped)

Credentials live in `/home/angkira/Project/software/cortex/.env.secrets`
(reference the **variable names**, never paste the values):

```bash
set -a; . /home/angkira/Project/software/cortex/.env.secrets; set +a
cd /home/angkira/Project/software/hars-longterm-memory

uv build   # or reuse dist/hars_longterm_memory-0.7.0-* from this branch

uv publish \
  --publish-url https://code.knotgate.com/api/v4/projects/14/packages/pypi \
  --username "$UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_USERNAME" \
  --password "$UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_PASSWORD"
```

- The token is the deploy token `UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_PASSWORD`
  (username `UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_USERNAME`). It must carry
  the `write_package_registry` scope; if it is read-only, create a
  `write_package_registry` token for this project instead.
- Never echo the token. Never commit `.env.secrets`.

## 4. Verify (do not trust the upload's own success line)

```bash
set -a; . /home/angkira/Project/software/cortex/.env.secrets; set +a

# PyPI simple index (readable with the deploy token):
curl -sS -u "$UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_USERNAME:$UV_INDEX_KNOTGATE_HARS_LONGTERM_MEMORY_PASSWORD" \
  "https://code.knotgate.com/api/v4/projects/14/packages/pypi/simple/hars-longterm-memory/" \
  | grep -o 'hars_longterm_memory-0.7.0[^"#]*'

# Or the Packages API (needs a token with read_api, not a deploy token):
curl -sS "https://code.knotgate.com/api/v4/projects/14/packages" \
  -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  | python3 -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
```

Expect `0.7.0` present, plus the container image tag `v0.7.0` in the GitLab
Container Registry (from `publish-container`).

---

## Blockers (beyond GitLab being down)

1. **In-flight agent work is uncommitted in the live tree.** `hars_memory/mcp_server.py`,
   `hars_memory/retrieval/fusion.py`, docs and several tests were being actively
   edited at preparation time (a running `opencode --auto` agent, cwd =
   this repo). Those changes are *not* part of `release/0.7.0`. Settle them
   first; then re-check whether they change the version decision (they add
   graph-branch routing / score-ordered context assembly, which are feature-ish).
   In particular the in-flight tree also **pins `lightrag-hku==1.5.7`**
   (`pyproject.toml` + `uv.lock`, audit 2026-10-17, see `docs/VERSION_PIN.md`) —
   that is a dependency change that MUST be in the release. It also means
   **`uv.lock` will conflict** between `release/0.7.0` and the agent's work:
   after merging, re-run `uv lock`, then re-run the tests and `uv build`, and
   re-check whether 0.7.0 is still the right version.
2. **`ruff check .` fails with 7 errors — pre-existing (identical at `v0.6.0`).**
   CI `lint` gates `publish-pypi`, so the tag pipeline will not publish until
   these are clean (or lint is bypassed):
   - `experiments/search-2026-10/gen_cases.py:448` F841
   - `experiments/search-2026-10/h1_rerank.py:297` F841
   - `experiments/search-2026-10/show.py:8` E702
   - `hars_memory/server/lightrag_init.py:605` F811 (duplicate `make_rerank_func_from_env`)
   - `tests/test_context_merge_budget.py:329` F401 (`json` unused)
   - `tests/test_embedder_http_mode.py:320` and `:530` F841 (unused `server`)
3. **One test fails on this host — pre-existing, environment-only.**
   `experiments/search-2026-10/ml/test_tokenize_russian.py::test_latin_identifier_multiset_matches_production`
   reads a hardcoded artifact `/tmp/search-exp/ml-clone/kv_store_text_chunks.json`
   that is absent. Full suite otherwise: **1159 passed, 27 skipped**. CI `test`
   gates `publish-pypi`; if the runner lacks that path the tag pipeline is
   blocked too (supply the artifact or skip that test on CI).

## Verification performed locally

| Check | Command | Result |
|---|---|---|
| Lock | `uv lock` | ok, `hars-longterm-memory 0.6.0 → 0.7.0` in `uv.lock` |
| Tests | `uv run pytest -q` | 1159 passed, 27 skipped, 1 failed (env-only, #3) |
| Lint | `uvx --from ruff==0.14.13 ruff check .` | 7 errors (pre-existing, #2) |
| Build | `uv build` | `dist/hars_longterm_memory-0.7.0-py3-none-any.whl` + `.tar.gz` |
