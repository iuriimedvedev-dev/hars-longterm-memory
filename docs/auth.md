# Authentication, projects and RBAC

Auth is **off by default** (`HARS_MEMORY_AUTH_ENABLED=0`): every caller has full
access, which is appropriate for a single-user local setup. Set it to `1` for any
shared deployment (MCP, gRPC).

## Keystore

Signing keys live in an encrypted keystore (AES-256-GCM), default
`~/.local/share/hars-longterm-memory/auth/keystore.enc` (`HARS_MEMORY_KEYSTORE_PATH`).
The master key comes from, in order:

1. `HARS_MEMORY_MASTER_KEY` (64-char hex, 44-char base64, or any string, which is SHA-256 hashed);
2. `HARS_MEMORY_AUTH_PASSPHRASE` (PBKDF2, 100k iterations);
3. an auto-generated key file `.../auth/master.key` (mode 0600).

Back up the master key separately from the keystore. Never commit either.

## Tokens (CLI)

```bash
uv run memory auth init-keys [--algorithm EdDSA|HS256] [--key-id ID] [--force]
uv run memory auth issue-token --user-id alice --roles developer \
    --dept sre --scopes "kb:read" --expires-in 90d
uv run memory auth inspect-token <jwt>
uv run memory auth revoke-token <jti-or-token>
uv run memory auth list-keys
```

`issue-token` options: `--user-id` (required), `--dept`, `--groups`, `--roles`,
`--scopes` (comma-separated, or `*`), `--expires-in` (`30d`, `90d`, `1y`,
`never`), `--jti`, `--json`. Default algorithm is EdDSA. Revocations are
hot-reloaded from disk.

Clients pass the token as the `access_token` tool argument, or via
`HARS_MEMORY_ACCESS_TOKEN`. Static (non-JWT) tokens can be configured with
`HARS_MEMORY_TOKENS_CONFIG` (a file path or inline JSON/YAML) or the default
`.../auth/tokens.json`; each entry carries `token`, `user_id`, `departments`,
`groups`, `roles`, `permissions`.

## Projects

A project isolates one knowledge base: its own index directory, staging
directory, BM25 cache and Qdrant collection prefix. Select it with the `project`
tool argument (default `default`).

Define projects with `HARS_MEMORY_PROJECTS_CONFIG` (file or inline YAML/JSON, either a
`projects:` list or a mapping of id to settings) or by placing directories under
`HARS_MEMORY_PROJECTS_DIR` (default `~/.local/share/hars-longterm-memory/projects`);
a subdirectory may hold `project.yaml`/`project.json`, otherwise `index/` and
`staging/` inside it are used. Settings: `name`, `index_dir`, `staging_dir`,
`sources_manifest`, `visibility` (`private`, `department`, `public`,
`shared_all`), `department`, `shared_departments`, `owner_user_id`,
`description`, `bm25_cache_dir`, `flat_dense_cache_dir`, `qdrant_collection`,
`qdrant_collection_prefix`.

## Permission rules

Actions are `read`, `write`, `admin`. With auth enabled, access is granted by the
first matching rule:

1. caller has the `admin` role;
2. caller is the project's `owner_user_id`;
3. a token permission matches `project:action` (project may be `*`; `admin`
   implies `write` and `read`; `write` implies `read`; `*` or `*:*` grants all);
4. department sharing: caller's departments/groups intersect the project's
   `shared_departments` (`viewer` role is read-only, others read/write);
5. project visibility `public` or `shared_all` allows `read` for any valid token.

No token means no access. Read tools need `read`; `memory_remember`, upsert/delete
need `write`; `memory_consolidate` and `memory_forget` need `admin`.
