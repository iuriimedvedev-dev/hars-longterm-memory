# Evaluation

Retrieval changes should be measured. The harness scores which **documents** a
retriever returns for a labelled query set.

## Metrics

`recall@k`, `nDCG@k`, MRR / first rank, supersession error rate (a superseded
document outranks the correct one), no-answer hit rate (hits returned for
questions the corpus cannot answer), file recall@k, and chunk coverage
(`hars_memory/eval/metrics.py`).

## Writing your own cases (no private data needed)

Create a small corpus and a query file. The repository ships a synthetic example
in `tests/e2e/fixtures/live_llm/` (`corpus/`, `queries.yaml`):

```yaml
queries:
  - id: owner
    type: conceptual        # identifier | conceptual | supersession | multihop | no_answer
    question: Who owns Aurora Relay and what is its routing code?
    gold_docs: [aurora.md]  # basenames, or path suffixes if the entry contains "/"
  - id: nothing
    type: no_answer
    question: What is the airspeed of an unladen swallow?
    gold_docs: []
```

Rules: `gold_docs` must be non-empty except for `no_answer`; for `supersession`
queries also list `superseded_docs`. Verify every gold entry by reading the
document, do not guess from file names. Keep queries free of confidential text if
you plan to publish the set.

## Running

```bash
# LLM-free pipeline (sparse | dense | fusion)
uv run memory build --paths tests/e2e/fixtures/live_llm/corpus --index-dir /tmp/eval-idx
uv run memory eval  --index-dir /tmp/eval-idx --queries tests/e2e/fixtures/live_llm/queries.yaml \
    --mode fusion --top-k 10 --report /tmp/eval-fusion.json

# Compare two reports; exit code 1 on regression (CI gate). Quality metrics and
# latency are both gated, so a slower candidate can fail even at equal quality.
uv run memory regress --baseline /tmp/eval-sparse.json --candidate /tmp/eval-fusion.json
```

## Strategy matrix

`memory strategy-bench --config matrix.yaml --run-id exp-001` builds each index
strategy once and runs every matching search strategy, repeating runs and
reporting latency and the primary metric. Start from
`config/strategy-bench.example.yaml` (chunk sizes, gleaning, storage backends,
`sparse`/`dense`/`fusion` search strategies).

## Tests

`uv run pytest -q` runs the offline suite (no LLM needed). Live-LLM end-to-end
tests are skipped unless `HARS_RUN_LIVE_LLM_E2E=1`; copy
`config/live-llm-e2e.env.example` to the git-ignored `config/live-llm-e2e.env`
and fill in your own endpoints.

## Other harnesses

`hars_memory/eval/` also contains `ab_bench.py` (A/B retrieval benchmark over a
labelled query file in the format above),
`check.py` (multi-hop gold-question checks with a question file you provide)
and `battle.py` (auto-generated "does the source text come back" cases).
The packaged default query files were removed from the public repository;
supply your own.
