"""Text chunker — splits content into overlapping character-level chunks.

Design constraints
------------------
- Pure Python, no LLM call.  This layer runs without the GPU.
- chunk_size and chunk_overlap come from config, never hardcoded.
- Returns a list of (chunk_text, chunk_index) tuples for easy tracing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk with its position metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source_id: str  # document-level stable ID


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split *text* into overlapping character-level chunks.

    Parameters
    ----------
    text:
        Full document text.
    source_id:
        Stable document ID — included in every chunk for provenance.
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Characters of overlap between consecutive chunks.

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

    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    idx = 0
    chunk_index = 0

    while idx < len(stripped):
        end = min(idx + chunk_size, len(stripped))
        chunk_text_str = stripped[idx:end]
        chunks.append(
            Chunk(
                text=chunk_text_str,
                chunk_index=chunk_index,
                start_char=idx,
                end_char=end,
                source_id=source_id,
            )
        )
        chunk_index += 1
        if end == len(stripped):
            break
        idx += stride

    return chunks
