# Security

## History rewrite: removal of a committed API key (2026-10-21)

A LiteLLM API key was committed to this repository in `config/live-llm-e2e.env` and
reached every branch and tag that existed at the time. The file was untracked from
`main` earlier, but the old blobs remained reachable through history, so the key
was still exposed to anyone who cloned or fetched the repository.

Because the key was in the object database (not merely in the working tree), the
only complete remediation was to rewrite history. On 2026-10-21 the history of this
GitHub mirror was rewritten with `git filter-repo --invert-paths --path
config/live-llm-e2e.env` and force-pushed. Three blobs that carried the leaked key
(`66d27b8`, `90cd030`, `a5ca356`) are no longer reachable from any ref.

### What was rewritten

- Repository: this GitHub mirror (`iuriimedvedev-dev/hars-longterm-memory`).
- Rewritten refs: `main`, `release/0.4.0`, `release/0.5.0`, the feature and fix
  branches, and all `v0.1.x`-`v0.8.1` tags.
- Commit SHAs changed for every commit that was an ancestor of the removed file.
  Tree contents are otherwise identical to the originals, minus
  `config/live-llm-e2e.env`.

### The JetBrains remote intentionally keeps the old history

The shared JetBrains remote (`origin`, `git.knotgate.com`) was **not** touched.
It still contains the original history, including the old `config/live-llm-e2e.env`
blobs. This is a deliberate split: the GitHub mirror is the clean, publishable
reference copy; `origin` remains the internal source of truth for the team.
Do not force-push this rewrite to `origin`.

### Residual risk is a dead string

The leaked key was **revoked and blocked** at the LiteLLM proxy. The new key is
different and is never committed (it lives in the operator's `~/.config/pier/env`).
The only residual risk from the old history on `origin` is a dead string that no
longer authenticates anything.

### Configuration is supplied out of band

This repository does not contain secrets. `config/live-llm-e2e.env.example`
documents the required variables with empty placeholders. Copy it to
`config/live-llm-e2e.env` and fill in your own values; that file is git-ignored.

### Syncing a stale clone

A clone or local checkout made before 2026-10-21 still has the old history and the
leaked blob. Because SHAs were rewritten, a normal `git pull` will not converge.
Reset to the rewritten branch instead:

```bash
git fetch github
git reset --hard github/main
```

For a clone that is only ever read, the simplest fix is to re-clone:

```bash
git clone git@github.com:iuriimedvedev-dev/hars-longterm-memory.git
```

### Reporting a leak

If you find credential material in this repository, report it through the team
channel and to the repository owner. Do not open a public pull request that
contains the material.
