"""Identifier-preserving tokenizer for BM25 sparse retrieval.

This corpus is queried with exact identifiers: `A2S32`, `phase_c1_lora_safe`,
`vea_native`, `hyp:7f62cfb4`. A tokenizer that stems, strips underscores/digits,
or splits inside alphanumeric codes destroys exactly the signal BM25 exists to
provide here (CORE-Bench / arXiv:2606.11864 names this failure mode explicitly
for dense retrieval; the whole point of adding a sparse channel is to not repeat
it in the sparse one too).

Design (see `tokenize_identifiers` for the full algorithm):

1. Extract identifier-shaped runs directly with a character-class regex
   (`[A-Za-z0-9_\\-:./]+`) rather than splitting on whitespace-then-stripping —
   this means punctuation like commas, parentheses, and quotes act as natural
   token boundaries for free, without needing a stopword/punctuation list.
2. Emit the **whole run** as one token, casefolded. This is what makes
   `A2S32` / `phase_c1_lora_safe` / `hyp:7f62cfb4` survive intact and match
   exactly at query time (query text goes through the identical function).
3. ALSO emit sub-tokens: split on `_ - : . /` delimiters, then split again on
   lowercase→uppercase (camelCase) boundaries. This lets a partial query like
   "lora safe" or "A2" still surface the chunk containing the full identifier,
   without which BM25's exact-term matching would be too brittle for queries
   that don't reproduce an identifier verbatim.
4. Casefold (not aggressive stemming/normalization) is applied identically to
   both indexed text and queries, so it does not lose information relative to
   an un-casefolded index — it only removes a distinction (upper vs lower)
   that would otherwise force a caller to reproduce a code's exact casing to
   find it, which callers of this MCP server routinely do not do.

Known, deliberate limitation: the camelCase sub-token split in step 3 uses
case information (a lowercase/digit -> uppercase transition), so it can only
fire on tokens that still carry that transition. "A2S32" splits into "a2" /
"s32"; an already-all-lowercase "a2s32" cannot — the transition it would be
based on does not exist in an all-lowercase string. This affects sub-token
*recall* only; the whole-token exact match (point 2, the primary guarantee
this module exists for) is unaffected and fully case-symmetric either way.

No stopword removal, no stemming: BM25's IDF term already down-weights common
words, and any stopword/stemmer list is exactly the kind of aggressive
normalization the identifier-retrieval evidence warns against. Kept out
deliberately, not as an oversight.
"""

from __future__ import annotations

import re

# Matches an identifier-shaped run: letters, digits, and the punctuation that
# commonly appears *inside* identifiers/paths/codes (underscore, hyphen, colon,
# dot, slash). Anything else (whitespace, commas, parens, quotes, ...) is a
# natural token boundary and is never included.
_IDENTIFIER_RUN_RE = re.compile(r"[A-Za-z0-9_\-:./]+")

# Delimiter characters that separate sub-tokens *inside* an identifier run,
# e.g. "phase_c1_lora_safe" -> "phase", "c1", "lora", "safe";
#      "hyp:7f62cfb4"       -> "hyp", "7f62cfb4".
_DELIMITER_RE = re.compile(r"[_\-:./]+")

# camelCase boundary: a lowercase letter or digit immediately followed by an
# uppercase letter, e.g. "VeaNative" -> "Vea", "Native"; "A2S32" -> "A2", "S32"
# (the digit '2' before 'S' counts as the lowercase-or-digit side).
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Characters that only carry meaning *between* other identifier characters,
# not at the edges of a run (e.g. a sentence-final period after a word, or a
# leading "--" on a CLI flag). Stripped only from the edges, never internally,
# so "e.g." -> "e.g" but "phase_c1_lora_safe" is untouched.
_EDGE_STRIP_CHARS = "_-:./"

# An identifier-shaped token: contains a colon-prefixed hex-ish suffix
# (`hyp:7f62cfb4`), an underscore (`phase_c1_lora_safe`), a camelCase boundary,
# or a mix of letters AND digits at least 3 chars long (`A2S32`). Used to
# decide when a query token should trigger the dedicated exact-identifier
# lookup path rather than relying on fusion weighting alone.
_PREFIXED_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*:[0-9A-Za-z]{4,}$")
_ALNUM_CODE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]{3,}$")


def _split_identifier(token: str) -> list[str]:
    """Split one identifier-shaped token into sub-tokens (delimiter + camelCase).

    Operates on the *original-case* token, since camelCase detection requires
    case information; callers casefold the resulting sub-tokens themselves.
    """
    parts: list[str] = []
    for delim_part in _DELIMITER_RE.split(token):
        if not delim_part:
            continue
        camel_parts = [p for p in _CAMEL_BOUNDARY_RE.split(delim_part) if p]
        parts.extend(camel_parts if camel_parts else [delim_part])
    return parts


def tokenize_identifiers(text: str) -> list[str]:
    """Tokenize `text` for BM25 indexing/querying, preserving whole identifiers.

    Returns a flat list of tokens (whole runs + their sub-tokens, casefolded,
    duplicates within a single run removed). Order is not significant to BM25
    but is kept deterministic for testability.
    """
    if not text:
        return []
    tokens: list[str] = []
    for raw_run in _IDENTIFIER_RUN_RE.findall(text):
        stripped = raw_run.strip(_EDGE_STRIP_CHARS)
        if not stripped:
            continue
        whole = stripped.casefold()
        tokens.append(whole)
        for sub in _split_identifier(stripped):
            sub_cf = sub.casefold()
            if sub_cf and sub_cf != whole:
                tokens.append(sub_cf)
    return tokens


def looks_like_identifier(token: str) -> bool:
    """True if `token` has the shape of a code identifier rather than a plain word.

    Matches: `hyp:7f62cfb4`-style prefixed IDs, `snake_case` names, camelCase
    names, and alphanumeric codes mixing letters and digits (`A2S32`). Used to
    decide when a query should trigger the dedicated exact-identifier lookup
    (see `retrieval.fusion`), not to gate normal tokenization.
    """
    if not token:
        return False
    if _PREFIXED_ID_RE.match(token):
        return True
    if "_" in token and len(token.strip("_")) > 0:
        return True
    if _CAMEL_BOUNDARY_RE.search(token) is not None:
        return True
    if _ALNUM_CODE_RE.match(token):
        return True
    return False


def extract_identifier_terms(text: str) -> list[str]:
    """Return the identifier-shaped runs found in `text`, original case, deduped
    in first-seen order. Used to detect "this question is asking about a
    specific code/entity" for the exact-identifier lookup path.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for raw_run in _IDENTIFIER_RUN_RE.findall(text):
        stripped = raw_run.strip(_EDGE_STRIP_CHARS)
        if not stripped or not looks_like_identifier(stripped):
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(stripped)
    return terms


__all__ = [
    "tokenize_identifiers",
    "looks_like_identifier",
    "extract_identifier_terms",
]
