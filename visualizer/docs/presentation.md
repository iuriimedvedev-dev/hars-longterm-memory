# Presentation mode — «Презентация: как работает память»

A guided 9-chapter walkthrough of how the HARS long-term memory works, driven
entirely by real data (the exported 47k-entity graph + one real `memory_recall`
trace). Launched from the tour button on the cosmos panel; `Esc` exits.

Bilingual: **EN (default)** and RU, switchable mid-presentation via the RU/EN
toggle (top-right) — the chapter position is kept, only the strings swap.
Language persists in `localStorage` (`mt.lang.v1`); a fresh browser (or a cleared
key) boots in English, and only an explicitly stored `ru` switches to Russian.
Both dictionaries are kept in exact parity (`Dict` is derived from the RU
literal and `EN: Dict` is an object literal — a key present in only one
dictionary is a build error). Presentation narration/facts are `L10n {ru, en}`
pairs in `lib/presentation.ts`; every panel string lives in `i18n/strings.ts`.

## Design rule — ILLUSTRATE, DON'T OCCUPY

The starfield is the hero. **Nothing in the presentation covers more than ~25%
of the viewport by default**, and nothing covers the centre region where the
camera is flying. Every chapter shows a small, translucent **schematic** that
illustrates its step; the full data (tables, per-channel contributions, row
movements, citations) is always **one click away**, never pre-expanded.

- a small persistent hint reads **"Click any element for details" /
  «Нажмите на элемент для деталей»** (bottom bar, in the active language);
- clicking a schematic element opens **ONE** detail card, docked at the right
  edge; it has an X and **`Esc` closes it first** (a second `Esc` exits the
  presentation);
- detail state resets on every chapter change.

## Controls

| Key / UI | Action |
| --- | --- |
| `←` / `→` | previous / next chapter |
| `Space` | play / pause the current chapter's animation |
| `F` | fullscreen |
| `Esc` | close a detail card / the chapter menu, or exit the presentation |
| «chapters» / «главы» button | chapter menu overlay |
| click any schematic element | open that chapter's ONE detail card |
| «данные этапа» button (gear icon) | opens the stage inspector over the live scene (trace chapters) |
| RU / EN toggle | switch narration language without losing position |

Every control glyph is a monochrome inline SVG (`components/icons.tsx`): the
presentation ships **zero emoji**, enforced by `npm run lint:no-emoji`
(also wired as `prebuild`, so `npm run build` fails on a pasted emoji).

## Chapter 2 (ingest) — compact schematic + two drill-downs

`components/IngestPanel.tsx`. The default view is ONE compact schematic
(≤ 340 px wide, docked top-left, translucent, ~5% of the viewport):

```
note chip ─▶ SECTIONS row (proportional blocks, labelled by heading)
          ─▶ CHUNKS row  (index windows, alternating tint, overlaps hatched)
5 chunks · 2 048 tok · overlap 64        <- from chunkingParams.ts
params: chunk_token_size 2048 · chunk_overlap_token_size 64 · …   (one line)
note → sections → chunks → teacher records
[Show document]  [Show 8 extracted records]
```

Every number is read from `lib/chunkingParams.ts` (`CHUNKING_PARAMS`, `paramRows`)
or the fixture — nothing is written in the component. Section blocks and window
blocks are positioned/width-ed proportionally over the same line axis; section
titles are printed only when the block is wide enough to hold them (the rest
carry a tooltip).

Two **collapsed-by-default** drill-downs (both in AnimatedCards):

- **Show document** → the full parameterised split preview: the editable
  parameter table, **both window lanes** over the shared line axis ("as
  ingested" = the verbatim index windows, and "md_struct, re-split from the
  parameters above" = the client-side mirror `splitNote()`), the note text
  tinted by the index windows with heading/section markers, overlap stripes and
  per-window token counts, plus the honest note about the two lanes. Clicking a
  window pins it and scrolls the note to its first line.
- **Show N extracted records** → the teacher's real extraction records
  (entities; relations when present), docked under the schematic. Hovering a
  window lights its records and dims the rest of the two panels, and vice versa.

Data: `scripts/export_ingest_fixture.py` (read-only) →
`app/public/data/ingest_fixture.json`: the note's full `doc.text` +
`doc.line_count` (32 227 chars / 1 074 lines), its five index windows
(2048/2048/2048/2048/308 tokens, real line overlaps at 186–217, 476–520,
760–793, 1035–1068) and the teacher's extraction records.

## Chapters 3–8 (transformations) — compact schematic strip + one detail card

`components/PresStageCards.tsx`. Each transformation chapter docks ONE small
card (≤ 260 px, `.topcard.compact`) at the top edge, semi-transparent, with a
one-line caption; the camera flight keeps playing behind it. Clicking an element
opens ONE detail card (the previous full tables) at the right edge.

| Chapter | Compact schematic (default) | Detail card (on click) |
| --- | --- | --- |
| 3 question→embedding | question chip → 768-d dimension strip; model/via; caption `{hits} dense hits · {ms} ms` | UMAP x,y,z strip, candidate pool, dense ms, top score, top-12 dense score bars |
| 4 channels | three channel pills: `vector N` · `bm25 N` · `graph N seeds` (one per detail) | per-channel tables: hits/top score/latency + bars (vector, BM25, graph) |
| 5 fusion | top-4 fused-score bars; meta `α (dense weight) = …`; caption `{n} chunks in the fused ranking` | fused bars + per-channel dense/sparse contribution bars with dense/sparse ranks |
| 6 rerank | before/after top-3 with moved rows highlighted; meta applied/skipped | before/after top-5, backend/pool/blocked inversions/moved count (or the honest skip reason) |
| 7 context | top-4 context blocks (width ∝ description length); meta = char budget | assembled chars / 15 000 budget, entity facts, relations, sources, preview quotes |
| 8 synthesis | answer quote chip; meta = client latency; caption `{n} citations` | answer quote + quotes/citations counts + citation list (`[A1]…`, `[ref n]`) |

Every number comes straight out of the loaded trace — never invented. Model
identity (EmbeddingGemma 768-d, Qwen3-Reranker-0.6B on iGPU) is a shared
constant in `app/src/lib/presentation.ts` (`PIPELINE`), the same facts the
narrations state, not per-trace telemetry. The 768-d row is a pure dimension
ruler (a CSS stripe), not fabricated per-dimension data — the trace carries the
real 3 UMAP coordinates, shown in the detail card.

## Chapters (viz source: `cosmos` = full 47k starfield, `trace` = the recall subgraph)

The narration texts below are the exact strings from
`app/src/lib/presentation.ts` (type-safe `L10n {ru, en}` pairs — a missing
translation is a build error). `{…}` are filled with real trace telemetry at
runtime.

### 1. Что это / What this is — `cosmos`
- **RU:** «Перед вами долговременная память агента: 1 854 заметки превращены в {nodes} сущностей и {edges} связей. Это GraphRAG-гибрид: векторный поиск по смыслу + knowledge graph + BM25 по словам. Каждая звезда — реальная сущность продакшн-индекса, положение — 3D UMAP по 768-мерным векторам.»
- **EN:** “This is an agent's long-term memory: 1,854 notes distilled into {nodes} entities and {edges} relations. A GraphRAG hybrid: semantic vector search + a knowledge graph + BM25 over words. Every star is a real entity of the production index; positions are a 3D UMAP of 768-dim vectors.”

### 2. Как знание попадает в память / How knowledge enters memory — `cosmos`
- Compact schematic + two drill-downs (above).

### 3. Вопрос → эмбеддинг / Question → embedding — `trace` (stage: query)
- The magenta query beacon lands at the question's true UMAP position.

### 4. Параллельные каналы / Parallel channels — `trace` (stage: vector)
- Cyan vector hits pulse in staggered by score; the narration contrasts the
  channels' failure modes.

### 5. Фьюжн / Fusion — `trace` (stage: fusion)
- Orange fused-ranking pulses; real α = 0.5; fact: `fusion recall@10 = 0.979 при 12.7 мс`.

### 6. Реранк / Rerank — `trace` (stage: rerank)
- White flares mark chunks that moved; guardrail/skip behaviour from telemetry.

### 7. Сборка контекста / Context assembly — `trace` (stage: context)
- Gold lines connect the query marker to every context entity; budget numbers
  (15 000 chars, 6 000 per doc, whole-document granularity).

### 8. Синтез / Synthesis — `trace` (stage: answer)
- The pink journey trail through query → fusion hits → answer anchors, camera
  pulled back; per-fact citations.

### 9. Цифры / The numbers — `cosmos`
- Measured results: `fusion recall@10 = 0.979 @ 12.7 мс; реранк ~1.6 с на iGPU;
  1 854 документа, 60 117 связей`.

## Screenshots (`docs/screenshots/`)

Captured headless (Chrome CDP, real-time waits — never
`--virtual-time-budget`, which zeroes canvases) against the built app:

- `pres-c2-compact-en.png` — chapter 2 default: compact schematic, detail
  collapsed, starfield visible and unobstructed behind it;
- `pres-c2-doc-en.png` / `pres-c2-doc-ru.png` — the document drill-down expanded
  (EN and RU);
- `pres-c2-records-en.png` — the extraction-records drill-down;
- `pres-c3-embed-en.png`, `pres-c4-channels-en.png`, `pres-c6-rerank-en.png`,
  `pres-c7-context-en.png`, `pres-c8-answer-en.png` — the compact strips;
- `pres-c5-strip-en.png` / `pres-c5-strip-ru.png` and `pres-c5-detail-en.png` —
  chapter 5 compact strip (EN/RU) and one expanded detail card;
- `app-cosmos-en.png`, `app-cosmos-legend-en.png`, `app-cosmos-ru.png`,
  `app-travel-en.png`, `app-travel-ru.png`, `app-about-ru.png`,
  `app-about-en.png` — the whole app in EN (default) and RU.

## Data provenance

- Full graph: `app/public/data/graph.json` (+3D UMAP coords) — exported from
  the production index, see `docs/SETUP.md` §3.
- Trace: `app/public/data/traces/trace_01.json` — compiled from a captured
  `memory_recall` fixture.
- Ingest fixture: `app/public/data/ingest_fixture.json` — verbatim index
  records (no synthesis): the note text, its index chunk windows and the
  teacher's extraction records.

## Zero-emoji gate

`npm run lint:no-emoji` (`scripts/no-emoji-check.mjs`, stdlib only) sweeps
`app/src` (ts/tsx/css/html), `app/index.html` and `app/public/data` JSON and
fails on any emoji/pictograph codepoint (emoji blocks, dingbats, misc symbols,
geometric shapes, flags, variation selectors). It runs as `prebuild`, so
`npm run build` cannot go green with an emoji in the tree. Typographic glyphs
the copy legitimately uses are not hits (arrows `← → ↑ ↓`, `· — – … « » × ≤ ≥ ≈ α §`);
in data fixtures the geometric-shapes block is allowed because it appears in
verbatim markdown ASCII-art diagrams (box drawing), while UI sources are
scanned with the full class. Replace a hit with an icon from
`app/src/components/icons.tsx`.

## Reduced motion / fallback

`prefers-reduced-motion` collapses the CSS animations globally (media query in
`index.css`); without WebGL2 the Canvas2D fallback renderer drives the same
chapters in flat projection (camera flights degrade to eased zooms). The compact
schematics are plain DOM/CSS, so they render identically in both renderers.
