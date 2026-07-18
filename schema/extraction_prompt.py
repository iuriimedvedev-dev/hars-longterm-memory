"""Extraction prompt seed for LightRAG.

``ENTITY_TYPES`` is injected into LightRAG's default extraction prompt via
``addon_params``.  ``DOMAIN_EXTRACTION_GUIDANCE`` is appended to LightRAG's
``PROMPTS["entity_extraction_system_prompt"]`` by ``lightrag_init.create_lightrag``
— naming normalisation is what keeps the graph from atomising into aliases.
"""

from __future__ import annotations

from tools.graphrag.schema.entity_types import EntityType, RelationType

# Comma-separated entity type list injected into LightRAG's extraction prompt.
ENTITY_TYPES: str = ", ".join(e.value for e in EntityType)

# Comma-separated relation type list.
RELATION_TYPES: str = ", ".join(r.value for r in RelationType)

# Appended verbatim to LightRAG's entity_extraction_system_prompt.
DOMAIN_EXTRACTION_GUIDANCE: str = f"""

---Domain Context (HARS robotics-ML project)---
The documents describe the HARS (Humanoid Action Reasoning System) project:
a VEA (Vision-Encoder Adapter) architecture trained on DROID robot data,
its hypotheses, experiments, checkpoints, metrics and infrastructure.

Preferred relation vocabulary (use as relationship keywords when they apply):
{RELATION_TYPES}

---Entity Naming Normalisation (CRITICAL)---
1. Experiment/hypothesis codes (A2S32, H6, B1, L1.1, E1, G0, Phase A/B/C):
   write them UPPERCASE exactly as printed, no added spaces or dots.
   Never merge different code families: "A2.5" (hypothesis variant) and
   "A2S5" (experiment session) are DIFFERENT entities — keep each verbatim.
2. Training runs named like vea_train_20260521_184235_c7981b: keep the full
   name verbatim as ONE entity; do not shorten or split it.
3. Database IDs/UUIDs: use stable prefixes — hyp:<id>, exp:<id>, run:<id>,
   ckpt:<basename>. If prose mentions both a code and its UUID, extract the
   code as the entity and the UUID form as a second entity related to it.
4. Models keep their full versioned names verbatim: "Qwen3.5-9B",
   "Gemma 4 26B", "DINOv3-large-336", "Depth-Anything-V3".
5. Always "Phase A" / "Phase B" / "Phase C" (capital P, space, capital letter).

---Markdown Tables---
Table rows and cells are data records, NOT entities. NEVER output a table row,
a delimiter row (|---|), or a cell fragment as an entity name. Read the table,
then extract the real-world entities and facts it describes.

---Type Discipline---
Use ONLY the listed entity types. If nothing fits, use type Other — do not
invent new types.
"""

# Backwards-compatible alias (tests import this name; contains hyp:/exp: markers).
EXTRACTION_SYSTEM_PROMPT: str = DOMAIN_EXTRACTION_GUIDANCE

__all__ = [
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "DOMAIN_EXTRACTION_GUIDANCE",
    "EXTRACTION_SYSTEM_PROMPT",
]
