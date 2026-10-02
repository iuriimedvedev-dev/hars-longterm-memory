# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
vulnerability reporting ("Security" tab, "Report a vulnerability") on this
repository, or contact the maintainer privately. Include affected versions,
reproduction steps and impact. We aim to acknowledge reports within a few days.

## Secrets

- Never commit API keys, tokens, master keys, passphrases or keystores. `.env`,
  `.env.*` (except `.env.example`) and `config/live-llm-e2e.env` are git-ignored;
  keep it that way.
- Use `HARS_MEMORY_LLM_API_KEY`, `HARS_MEMORY_MASTER_KEY` and similar variables
  from a secret manager or an untracked env file.
- If a secret is ever committed, revoke it immediately; rewriting history is not
  enough.

## Deployment notes

- Authentication is off by default (`HARS_MEMORY_AUTH_ENABLED=0`, full access for
  every caller). Enable it for any shared or networked deployment; see
  `docs/auth.md`.
- The gRPC server binds `0.0.0.0` by default (`HARS_MEMORY_GRPC_HOST`); restrict it
  or put it behind a trusted network or proxy when auth is disabled.
- The index may contain everything you indexed. Protect index directories and
  exported snapshots like the source documents.
