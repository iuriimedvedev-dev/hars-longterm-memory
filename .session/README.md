# `.session/` — work-session log

One markdown file per significant work session, named:

```
.session/YYYY-MM-DD_topic.md
```

Each entry records, in compact English:

- **Decisions made** — what was chosen and, briefly, why.
- **Changes shipped** — what actually landed (files, behavior, version bumps).
- **Next steps** — what's left, what's deliberately deferred, what to check
  next time.

This mirrors the convention already used in the parent `cortex` monorepo this
package was extracted from. It exists so a future agent (or the user) can
reconstruct *why* something is the way it is without re-deriving it purely
from `git log` — a commit message explains one change; a session file
explains the reasoning and context around a whole work session, including
things that were tried and rejected.

Keep entries factual and grounded — record what was actually verified, not
what was merely intended or assumed to be true.
