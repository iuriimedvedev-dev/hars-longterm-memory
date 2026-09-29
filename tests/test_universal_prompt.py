"""Universal extraction prompt: core vocabulary + optional, escaped project additions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from hars_memory.schema.loader import DEFAULT_SCHEMA_PATH, Schema, load_schema
from hars_memory.schema.universal_prompt import (
    CORE_ENTITY_TYPES,
    CORE_RELATION_TYPES,
    apply_to_lightrag_prompts,
    build_examples,
    build_system_prompt,
    effective_entity_types,
    effective_relation_types,
)

TUPLE = "<|#|>"
COMPLETE = "<|COMPLETE|>"


def _schema(**kw) -> Schema:
    base = dict(entity_types=[], relation_types=[], guidance="")
    base.update(kw)
    return Schema(**base)


def _render(system: str, examples: list[str], types: list[str]) -> str:
    """Mirror lightrag.operate: format examples, then the system prompt."""
    ex = "\n".join(examples).format(
        tuple_delimiter=TUPLE, completion_delimiter=COMPLETE, entity_types=", ".join(types), language="English"
    )
    return system.format(
        tuple_delimiter=TUPLE, completion_delimiter=COMPLETE, entity_types=",".join(types), examples=ex,
        language="English",
    )


def test_core_only_prompt_renders_through_lightrag_format() -> None:
    s = _schema()
    out = _render(build_system_prompt(s), build_examples(s), effective_entity_types(s))
    assert "{" not in out.replace(TUPLE, "").replace(COMPLETE, "")
    assert "Project Guidance" not in out
    for t in CORE_ENTITY_TYPES:
        assert t in out


def test_project_additions_are_merged_and_deduplicated() -> None:
    s = _schema(entity_types=["Hypothesis", "Concept"], relation_types=["tests", "uses"])
    assert effective_entity_types(s) == [*CORE_ENTITY_TYPES, "Hypothesis"]
    rels = effective_relation_types(s)
    assert rels == [*CORE_RELATION_TYPES, "tests"]
    assert "tests" in build_system_prompt(s)


def test_include_core_false_uses_project_vocabulary_only() -> None:
    s = _schema(entity_types=["Hypothesis"], relation_types=["tests"], include_core=False)
    assert effective_entity_types(s) == ["Hypothesis"]
    assert build_examples(s) == []


def test_guidance_with_braces_cannot_break_lightrag_format() -> None:
    s = _schema(guidance='Use ids like {id} and JSON {"a": 1}; keep C{x}.')
    out = _render(build_system_prompt(s), build_examples(s), effective_entity_types(s))
    assert '{id}' in out and '{"a": 1}' in out
    assert "---Project Guidance---" in out


def test_project_examples_accept_delimiter_placeholders_and_escape_the_rest() -> None:
    ex = "entity{tuple_delimiter}X{tuple_delimiter}Tool{tuple_delimiter}uses {curly} braces\n{completion_delimiter}"
    s = _schema(examples=[ex])
    out = _render(build_system_prompt(s), build_examples(s), effective_entity_types(s))
    assert f"entity{TUPLE}X{TUPLE}Tool{TUPLE}uses {{curly}} braces" in out


def test_builtin_examples_only_use_allowed_types_and_valid_field_counts() -> None:
    s = _schema()
    types = set(effective_entity_types(s))
    for ex in build_examples(s):
        body = ex.split("<Output>")[1].format(tuple_delimiter=TUPLE, completion_delimiter=COMPLETE)
        names = set()
        for line in body.splitlines():
            if line.startswith("entity" + TUPLE):
                f = line.split(TUPLE)
                assert len(f) == 4 and f[2] in types, line
                names.add(f[1])
            elif line.startswith("relation" + TUPLE):
                f = line.split(TUPLE)
                assert len(f) == 5, line
                assert f[1] in names and f[2] in names, f"dangling endpoint: {line}"


def test_apply_replaces_and_is_idempotent() -> None:
    prompts = {"entity_extraction_system_prompt": "stock", "entity_extraction_examples": ["stock"]}
    s = _schema(guidance="G1")
    apply_to_lightrag_prompts(prompts, s)
    first = dict(prompts)
    apply_to_lightrag_prompts(prompts, s)
    assert prompts == first
    assert "stock" not in prompts["entity_extraction_system_prompt"]
    assert prompts["entity_extraction_system_prompt"].count("G1") == 1


def test_load_schema_reads_optional_keys(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.dump({"entity_types": ["A"], "relation_types": ["r"], "guidance": "g",
                            "examples": ["e1"], "include_core": False}))
    s = load_schema(p)
    assert s.examples == ["e1"] and s.include_core is False


def test_default_schema_stays_generic_and_backward_compatible() -> None:
    s = load_schema(DEFAULT_SCHEMA_PATH)
    assert s.include_core is True and s.examples == []


def test_cortex_style_schema_extends_core() -> None:
    s = _schema(entity_types=["Hypothesis", "Experiment"], guidance="Codes are UPPERCASE.")
    types = effective_entity_types(s)
    assert "Hypothesis" in types and "Document" in types
    assert re.search(r"Codes are UPPERCASE", build_system_prompt(s))


def test_llm_extra_body_env_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    from hars_memory.server import lightrag_init

    monkeypatch.setenv("HARS_MEMORY_LLM_EXTRA_BODY", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        lightrag_init.make_llm_func("http://x/v1", "m", 128, 0.0)
    monkeypatch.setenv("HARS_MEMORY_LLM_EXTRA_BODY", "{bad")
    with pytest.raises(ValueError, match="not valid JSON"):
        lightrag_init.make_llm_func("http://x/v1", "m", 128, 0.0)
