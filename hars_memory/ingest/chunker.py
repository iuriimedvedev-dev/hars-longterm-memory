"""Text chunker — Markdown-aware splitting with a character-window fallback.

Design constraints
------------------
- Pure Python, no LLM call.  This layer runs without the GPU.
- chunk_size and chunk_overlap come from config, never hardcoded.
- Returns a list of ``Chunk`` records for easy tracing.

Why Markdown-aware
------------------
A blind character window cuts documents mid-sentence, mid-code-fence and
mid-table, and — worse for retrieval — strips every chunk of the heading it
lived under.  A chunk reading "set ``replicaCount: 3``" is unrankable and
unusable without "## Scaling the ingress controller" above it.  So for content
that actually looks like Markdown (an ATX heading exists outside a code
fence) we split on H2/H3 boundaries and prepend the heading breadcrumb
(``# Doc > ## Section > ### Subsection``) to every chunk produced from that
section.

``chunk_size`` remains a hard upper bound: an oversized section is broken at
block boundaries (blank-line-separated paragraphs, with fenced code blocks and
pipe tables treated as indivisible units), and only a single block that is
itself larger than the budget falls back to the overlapping character window.

Non-Markdown input (plain prose, JSON, source code) keeps the original
overlapping character-window behaviour byte-for-byte — callers relying on that
(``corpus/build.py``, ``service/engines.py``) see no change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ATX headings only (``## Foo``).  Setext (``Foo\n---``) is deliberately not
# recognised: ``---`` is also the YAML-frontmatter fence every walked KB file
# opens with, so treating it as a heading underline would mis-split the very
# documents this feature exists for.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|")

# Section boundaries.  H1 is excluded on purpose: it is the document title, so
# breaking on it yields one chunk per document and defeats the split.
_SPLIT_LEVELS = (2, 3)

_BREADCRUMB_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk with its position metadata.

    ``start_char``/``end_char`` always address the *source body span* inside
    the stripped input text.  For a Markdown chunk, ``text`` additionally
    carries a prepended heading breadcrumb, so ``text`` is not necessarily
    ``source[start_char:end_char]``; ``breadcrumb`` records exactly what was
    prepended (empty string when nothing was).
    """

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source_id: str  # document-level stable ID
    breadcrumb: str = ""


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    markdown_aware: bool = True,
) -> list[Chunk]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Parameters
    ----------
    text:
        Full document text.
    source_id:
        Stable document ID — included in every chunk for provenance.
    chunk_size:
        Maximum characters per chunk, breadcrumb included.
    chunk_overlap:
        Characters of overlap between consecutive chunks.  Applies to the
        character-window path only: Markdown sections and blocks are split at
        semantic boundaries, where duplicated context buys nothing.
    markdown_aware:
        When True (default) and *text* contains an ATX heading outside a code
        fence, split on H2/H3 boundaries and prepend heading breadcrumbs.
        Set False to force the plain character window.

    Returns
    -------
    list[Chunk]
        Empty list when text is empty or whitespace-only.

    Raises
    ------
    ValueError
        When chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    stripped = text.strip()
    if not stripped:
        return []

    if markdown_aware and _looks_like_markdown(stripped):
        return _chunk_markdown(stripped, source_id, chunk_size, chunk_overlap)
    return _chunk_window(stripped, source_id, chunk_size, chunk_overlap)


# --- plain character window (original behaviour) ----------------------------


def _chunk_window(
    stripped: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    base_offset: int = 0,
    first_index: int = 0,
    breadcrumb: str = "",
) -> list[Chunk]:
    """Overlapping character window over *stripped*.

    *breadcrumb*, when given, is prepended to every emitted chunk and its
    length is subtracted from the per-chunk character budget so the declared
    ``chunk_size`` stays a real upper bound on ``len(chunk.text)``.
    """
    prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
    budget = chunk_size - len(prefix)
    if budget <= chunk_overlap:
        # Pathological: the breadcrumb eats the whole budget.  Drop the
        # overlap rather than loop forever or emit empty chunks.
        budget = max(1, chunk_size - len(prefix))
        stride = budget
    else:
        stride = budget - chunk_overlap

    chunks: list[Chunk] = []
    idx = 0
    chunk_index = first_index
    while idx < len(stripped):
        end = min(idx + budget, len(stripped))
        chunks.append(
            Chunk(
                text=prefix + stripped[idx:end],
                chunk_index=chunk_index,
                start_char=base_offset + idx,
                end_char=base_offset + end,
                source_id=source_id,
                breadcrumb=breadcrumb,
            )
        )
        chunk_index += 1
        if end == len(stripped):
            break
        idx += stride
    return chunks


# --- Markdown-aware path ----------------------------------------------------


def _looks_like_markdown(stripped: str) -> bool:
    """True iff at least one ATX heading exists outside a fenced code block."""
    in_fence = False
    for line in stripped.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and _HEADING_RE.match(line):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _Section:
    breadcrumb: str
    body: str
    start_char: int


def _split_sections(stripped: str) -> list[_Section]:
    """Split *stripped* at H2/H3 headings, carrying the heading breadcrumb.

    Every section owns the heading line that opens it (so the heading text is
    indexed with its content), and the breadcrumb repeats the full ancestor
    chain including that heading — a chunk cut out of the middle of a long
    section is otherwise indistinguishable from one cut out of any other.
    """
    lines = stripped.splitlines(keepends=True)
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, raw heading text)
    current: list[str] = []
    current_start = 0
    current_breadcrumb = ""
    offset = 0
    in_fence = False

    def flush() -> None:
        if not current:
            return
        body = "".join(current)
        if body.strip():
            sections.append(
                _Section(
                    breadcrumb=current_breadcrumb,
                    body=body.strip(),
                    start_char=current_start + (len(body) - len(body.lstrip())),
                )
            )
        current.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            offset += len(line)
            continue

        heading = None if in_fence else _HEADING_RE.match(line.rstrip("\n"))
        if heading is not None:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line.strip()))
            if level in _SPLIT_LEVELS:
                flush()
                current_start = offset
                current_breadcrumb = _BREADCRUMB_SEPARATOR.join(
                    raw for _, raw in stack
                )
        current.append(line)
        offset += len(line)

    flush()
    return sections


def _split_blocks(body: str) -> list[tuple[int, str]]:
    """Split a section body into ``(offset_in_body, block_text)`` units.

    A block is a blank-line-separated paragraph, EXCEPT that a fenced code
    block and a run of pipe-table rows are each kept whole regardless of the
    blank lines inside them — splitting a code fence or a table mid-way
    produces syntactically broken, unusable context.
    """
    lines = body.splitlines(keepends=True)
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    current_start = 0
    offset = 0
    in_fence = False

    def flush() -> None:
        nonlocal current_start
        if current and "".join(current).strip():
            blocks.append((current_start, "".join(current).strip()))
        current.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            if not in_fence:
                flush()
                current_start = offset
            in_fence = not in_fence
            current.append(line)
            offset += len(line)
            if not in_fence:
                flush()
                current_start = offset
            continue

        if in_fence:
            current.append(line)
            offset += len(line)
            continue

        if not line.strip():
            # A blank line inside a table run is still a table break, but a
            # blank line anywhere else closes the paragraph.
            flush()
            offset += len(line)
            current_start = offset
            continue

        if _TABLE_ROW_RE.match(line) and current and not _TABLE_ROW_RE.match(current[0]):
            # Table starts right after prose with no blank line between.
            flush()
            current_start = offset
        current.append(line)
        offset += len(line)

    flush()
    return blocks


def _chunk_markdown(
    stripped: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_sections(stripped):
        prefix_len = len(section.breadcrumb) + 2 if section.breadcrumb else 0
        if prefix_len + len(section.body) <= chunk_size:
            chunks.append(
                _make_chunk(
                    section.breadcrumb,
                    section.body,
                    len(chunks),
                    section.start_char,
                    section.start_char + len(section.body),
                    source_id,
                )
            )
            continue

        # Oversized section: pack whole blocks up to the budget (soft break).
        budget = max(1, chunk_size - prefix_len)
        pending: list[tuple[int, str]] = []
        pending_len = 0
        for block_offset, block in _split_blocks(section.body):
            if len(block) > budget:
                chunks.extend(
                    _flush_pending(
                        pending, section, source_id, len(chunks)
                    )
                )
                pending, pending_len = [], 0
                chunks.extend(
                    _chunk_window(
                        block,
                        source_id,
                        chunk_size,
                        chunk_overlap,
                        base_offset=section.start_char + block_offset,
                        first_index=len(chunks),
                        breadcrumb=section.breadcrumb,
                    )
                )
                continue
            projected = pending_len + (2 if pending else 0) + len(block)
            if pending and projected > budget:
                chunks.extend(
                    _flush_pending(pending, section, source_id, len(chunks))
                )
                pending, pending_len = [], 0
                projected = len(block)
            pending.append((block_offset, block))
            pending_len = projected
        chunks.extend(_flush_pending(pending, section, source_id, len(chunks)))

    return chunks


def _flush_pending(
    pending: list[tuple[int, str]],
    section: _Section,
    source_id: str,
    chunk_index: int,
) -> list[Chunk]:
    if not pending:
        return []
    body = "\n\n".join(block for _, block in pending)
    start = section.start_char + pending[0][0]
    end = section.start_char + pending[-1][0] + len(pending[-1][1])
    return [
        _make_chunk(
            section.breadcrumb, body, chunk_index, start, end, source_id
        )
    ]


def _make_chunk(
    breadcrumb: str,
    body: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    source_id: str,
) -> Chunk:
    text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
    return Chunk(
        text=text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        source_id=source_id,
        breadcrumb=breadcrumb,
    )


__all__ = ["Chunk", "chunk_text"]
