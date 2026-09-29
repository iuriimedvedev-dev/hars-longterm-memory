"""Universal, domain-agnostic entity/relation extraction prompt for LightRAG.

LightRAG's stock extraction prompt ships three worked examples about fiction,
stock markets and sports, whose lowercase types (``equipment``, ``category``)
contradict a project's own type list, and it has no rules for identifiers or
Markdown tables. This module replaces it with a compact prompt that:

* is domain-neutral out of the box (core entity + relation vocabulary);
* keeps LightRAG's exact output format, so its parser is unchanged;
* accepts *optional*, additive project configuration from the schema YAML
  (``hars_memory.schema.loader.Schema``): extra entity types, extra relation
  vocabulary, free-text ``guidance`` and extra worked ``examples``.

Config text is escaped before it is spliced into the template, because
LightRAG runs ``str.format`` on the prompt: a literal ``{`` in a project's
guidance would otherwise crash extraction.
"""

from __future__ import annotations

from typing import Any, Final, MutableMapping

from hars_memory.schema.loader import Schema

#: Always allowed; project types are appended to these (unless ``include_core: false``).
CORE_ENTITY_TYPES: Final[tuple[str, ...]] = (
    "Concept",
    "Document",
    "Decision",
    "Event",
    "Person",
    "Organization",
    "Tool",
    "Component",
    "Task",
)

#: Preferred relationship keywords; project relations are appended.
CORE_RELATION_TYPES: Final[tuple[str, ...]] = (
    "relates_to",
    "part_of",
    "depends_on",
    "uses",
    "produces",
    "derived_from",
    "supersedes",
    "documented_in",
)

_RELATION_TOKEN: Final = "<<RELATION_TYPES>>"
_GUIDANCE_TOKEN: Final = "<<PROJECT_GUIDANCE>>"

# Placeholders {entity_types} {tuple_delimiter} {completion_delimiter}
# {language} {examples} are filled by LightRAG at extraction time.
UNIVERSAL_SYSTEM_PROMPT: Final = """---Role---
You are a knowledge-graph extraction specialist. Turn the input text into entities and binary relationships.

---Instructions---
1. Entities. For each meaningful entity output one line with 4 fields separated by `{tuple_delimiter}`:
   `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
   * entity_type: exactly one of `{entity_types}`. If none fits, use `Other`; never invent a type.
   * entity_name: identifiers are copied VERBATIM, keeping case, digits, dots, dashes and underscores: experiment/ticket/issue codes, versions, model and library names, file paths, commands, config keys, UUIDs and hashes. Never shorten, split, re-case or merge different identifiers. Other names use their natural form, in one consistent spelling across the whole output.
   * entity_description: concise, third person, based only on the text. Name the subject explicitly; no pronouns and no "this document/article".
   * Skip generic words, section headings, bare dates and numbers with no subject.
2. Markdown tables. Rows, cells and `|---|` separators are data, not entities. Never output a row or a cell fragment as an entity name; extract the real things the table describes (the row subject, its metrics, its status).
3. Relationships. For each direct, clearly stated relationship between two extracted entities output one line with 5 fields separated by `{tuple_delimiter}`:
   `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`
   * source and target must match an extracted entity_name exactly.
   * relationship_keywords: comma-separated. Prefer this vocabulary when it applies: <<RELATION_TYPES>>. Otherwise use one to three short lowercase snake_case keywords. Never use `{tuple_delimiter}` inside the keywords.
   * Decompose N-ary statements into binary pairs. Relationships are undirected unless the text states a direction. No duplicates.
4. `{tuple_delimiter}` is an atomic field separator and must never contain content.
5. Output all entities first, then relationships, the most significant first. Write everything in `{language}`; proper nouns keep their original spelling.
6. Output only the list, then the literal `{completion_delimiter}` on its own final line.
<<PROJECT_GUIDANCE>>
---Examples---
{examples}
"""

_EXAMPLE_INCIDENT: Final = """<Entity_types>
["Concept","Document","Decision","Event","Person","Organization","Tool","Component","Task"]

<Input Text>
```
2026-03-14 - INC-2043: the payments-api pod restarted every few minutes after release v2.8.1.
Maria Chen traced it to the connection pool in db/pool.py (POOL_MAX_SIZE=5) and the team decided to raise it to 20 and roll back to v2.8.0 meanwhile. Runbook: docs/runbooks/db-pool.md.
```

<Output>
entity{tuple_delimiter}INC-2043{tuple_delimiter}Event{tuple_delimiter}INC-2043 is an incident on 2026-03-14 in which the payments-api pod restarted every few minutes after release v2.8.1.
entity{tuple_delimiter}payments-api{tuple_delimiter}Component{tuple_delimiter}payments-api is a service whose pod restarted repeatedly after release v2.8.1.
entity{tuple_delimiter}v2.8.1{tuple_delimiter}Concept{tuple_delimiter}v2.8.1 is the payments-api release after which the restarts began.
entity{tuple_delimiter}v2.8.0{tuple_delimiter}Concept{tuple_delimiter}v2.8.0 is the earlier payments-api release chosen as the temporary rollback target.
entity{tuple_delimiter}Maria Chen{tuple_delimiter}Person{tuple_delimiter}Maria Chen traced INC-2043 to the database connection pool configuration.
entity{tuple_delimiter}db/pool.py{tuple_delimiter}Component{tuple_delimiter}db/pool.py contains the connection pool setting POOL_MAX_SIZE=5 that caused the restarts.
entity{tuple_delimiter}Raise POOL_MAX_SIZE to 20{tuple_delimiter}Decision{tuple_delimiter}The team decided to raise POOL_MAX_SIZE from 5 to 20 and roll back to v2.8.0 in the meantime.
entity{tuple_delimiter}docs/runbooks/db-pool.md{tuple_delimiter}Document{tuple_delimiter}docs/runbooks/db-pool.md is the runbook covering the database connection pool.
relation{tuple_delimiter}INC-2043{tuple_delimiter}payments-api{tuple_delimiter}affects{tuple_delimiter}INC-2043 is an incident affecting the payments-api service.
relation{tuple_delimiter}INC-2043{tuple_delimiter}v2.8.1{tuple_delimiter}derived_from{tuple_delimiter}INC-2043 began after release v2.8.1 was deployed.
relation{tuple_delimiter}Maria Chen{tuple_delimiter}db/pool.py{tuple_delimiter}relates_to, root_cause{tuple_delimiter}Maria Chen identified db/pool.py as the root cause of INC-2043.
relation{tuple_delimiter}Raise POOL_MAX_SIZE to 20{tuple_delimiter}INC-2043{tuple_delimiter}relates_to, mitigates{tuple_delimiter}Raising POOL_MAX_SIZE to 20 is the decision taken to resolve INC-2043.
relation{tuple_delimiter}Raise POOL_MAX_SIZE to 20{tuple_delimiter}v2.8.0{tuple_delimiter}uses{tuple_delimiter}The decision uses a rollback to v2.8.0 as an interim measure.
relation{tuple_delimiter}INC-2043{tuple_delimiter}docs/runbooks/db-pool.md{tuple_delimiter}documented_in{tuple_delimiter}INC-2043 is covered by the db-pool runbook.
{completion_delimiter}
"""

_EXAMPLE_TABLE: Final = """<Entity_types>
["Concept","Document","Decision","Event","Person","Organization","Tool","Component","Task"]

<Input Text>
```
## Benchmark results
| Run | Model | F1 | Status |
|---|---|---|---|
| R-17 | bge-m3 | 0.81 | baseline |
| R-18 | e5-large-v2 | 0.86 | supersedes R-17 |
```

<Output>
entity{tuple_delimiter}R-17{tuple_delimiter}Event{tuple_delimiter}R-17 is a benchmark run of bge-m3 with F1 0.81 that serves as the baseline.
entity{tuple_delimiter}R-18{tuple_delimiter}Event{tuple_delimiter}R-18 is a benchmark run of e5-large-v2 with F1 0.86 that supersedes R-17.
entity{tuple_delimiter}bge-m3{tuple_delimiter}Tool{tuple_delimiter}bge-m3 is an embedding model evaluated in run R-17.
entity{tuple_delimiter}e5-large-v2{tuple_delimiter}Tool{tuple_delimiter}e5-large-v2 is an embedding model evaluated in run R-18 with a higher F1 than the baseline.
relation{tuple_delimiter}R-18{tuple_delimiter}R-17{tuple_delimiter}supersedes{tuple_delimiter}R-18 supersedes the baseline run R-17 with F1 0.86 versus 0.81.
relation{tuple_delimiter}R-17{tuple_delimiter}bge-m3{tuple_delimiter}uses{tuple_delimiter}R-17 evaluated bge-m3.
relation{tuple_delimiter}R-18{tuple_delimiter}e5-large-v2{tuple_delimiter}uses{tuple_delimiter}R-18 evaluated e5-large-v2.
{completion_delimiter}
"""

CORE_EXAMPLES: Final[tuple[str, ...]] = (_EXAMPLE_INCIDENT, _EXAMPLE_TABLE)


def _escape_braces(text: str) -> str:
    """Make config text inert for the ``str.format`` LightRAG applies later."""
    return text.replace("{", "{{").replace("}", "}}")


def _merged(core: tuple[str, ...], extra: list[str], include_core: bool) -> list[str]:
    ordered = [*core, *extra] if include_core else list(extra)
    seen: set[str] = set()
    return [v for v in ordered if not (v in seen or seen.add(v))]  # type: ignore[func-returns-value]


def effective_entity_types(schema: Schema) -> list[str]:
    """Core types plus the project's own (deduplicated, order preserved)."""
    return _merged(CORE_ENTITY_TYPES, schema.entity_types, schema.include_core)


def effective_relation_types(schema: Schema) -> list[str]:
    return _merged(CORE_RELATION_TYPES, schema.relation_types, schema.include_core)


def build_system_prompt(schema: Schema) -> str:
    """Render the universal system prompt with the project's optional additions."""
    guidance = schema.guidance.strip()
    block = f"\n---Project Guidance---\n{_escape_braces(guidance)}\n" if guidance else ""
    return UNIVERSAL_SYSTEM_PROMPT.replace(
        _RELATION_TOKEN, _escape_braces(", ".join(effective_relation_types(schema)))
    ).replace(_GUIDANCE_TOKEN, block)


def build_examples(schema: Schema) -> list[str]:
    """Built-in examples plus the project's own.

    Project examples may use ``{tuple_delimiter}`` / ``{completion_delimiter}``
    placeholders like the built-ins; any other brace is escaped.
    """
    extra = []
    for text in schema.examples:
        safe = _escape_braces(text)
        for name in ("tuple_delimiter", "completion_delimiter"):
            safe = safe.replace("{{" + name + "}}", "{" + name + "}")
        extra.append(safe)
    core = list(CORE_EXAMPLES) if schema.include_core else []
    return [*core, *extra]


def apply_to_lightrag_prompts(prompts: MutableMapping[str, Any], schema: Schema) -> None:
    """Install the universal prompt into ``lightrag.prompt.PROMPTS``.

    Replaces (never appends), so it is idempotent. LightRAG reads ``PROMPTS``
    globally at extraction time, hence the schema is per *process*: run one
    indexing process per project.
    """
    prompts["entity_extraction_system_prompt"] = build_system_prompt(schema)
    prompts["entity_extraction_examples"] = build_examples(schema)


__all__ = [
    "CORE_ENTITY_TYPES",
    "CORE_RELATION_TYPES",
    "CORE_EXAMPLES",
    "UNIVERSAL_SYSTEM_PROMPT",
    "apply_to_lightrag_prompts",
    "build_examples",
    "build_system_prompt",
    "effective_entity_types",
    "effective_relation_types",
]
