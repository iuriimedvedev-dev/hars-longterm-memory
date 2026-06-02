"""Extraction prompt seed for LightRAG.

LightRAG injects this ENTITY_TYPES string into its default extraction prompt
so the LLM knows which entity and relation types to look for.
"""

from __future__ import annotations

from tools.graphrag.schema.entity_types import EntityType, RelationType

# Comma-separated entity type list injected into LightRAG's extraction prompt.
ENTITY_TYPES: str = ", ".join(e.value for e in EntityType)

# Comma-separated relation type list.
RELATION_TYPES: str = ", ".join(r.value for r in RelationType)

# Full extraction-prompt seed text (injected via LightRAG's custom_prompt_func).
EXTRACTION_SYSTEM_PROMPT: str = f"""You are extracting a structured knowledge graph from robotics-ML project
documents.  The project is called HARS (Humanoid Action Reasoning System) and
uses a VEA (Vision-Encoder Adapter) architecture trained on DROID robot data.

## Entity types to extract
{ENTITY_TYPES}

## Relation types to extract
{RELATION_TYPES}

## Stable IDs
When you encounter references to database objects use the following ID prefixes:
- Hypothesis → hyp:<id>   (e.g. hyp:h6, hyp:abc-123-uuid)
- Experiment → exp:<id>   (e.g. exp:42, exp:uuid)
- Checkpoint → ckpt:<hash-or-path-fragment>
- JobRun     → run:<id>

## Rules
1. Extract ONLY entity types and relation types listed above.
2. For each triple output: (subject_name, relation_type, object_name).
3. Normalise entity names to canonical forms (e.g. "Phase C" not "phaseC").
4. Do NOT invent facts not present in the text.
5. Use the stable-ID prefix if the text contains an explicit ID or UUID.
"""

__all__ = ["ENTITY_TYPES", "RELATION_TYPES", "EXTRACTION_SYSTEM_PROMPT"]
