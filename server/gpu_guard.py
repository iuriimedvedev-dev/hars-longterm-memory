"""GPU concurrency guard — mirrors .claude/skills/distillation.py pattern.

Indexing requires the Qwen3.6-27B extractor LLM which is GPU-exclusive.
This module provides the same check used by distillation.py so indexing
automatically blocks when vea/expert/ai_tuner/finetune/distillation is running.

Query path NEVER calls this guard — CPU embeddings keep queries always available.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_GPU_EXCLUSIVE_WORKFLOWS: frozenset[str] = frozenset(
    {"vea", "expert", "ai_tuner", "finetune", "distillation"}
)


class GpuBusyError(RuntimeError):
    """Raised when a GPU-exclusive workflow is running."""

    def __init__(self, blocking_id: str, workflow: str) -> None:
        super().__init__(
            f"GPU is busy: workflow '{workflow}' (experiment {blocking_id}) is running. "
            "Indexing requires the Qwen3.6-27B extractor — wait for GPU to be free."
        )
        self.blocking_id = blocking_id
        self.workflow = workflow


def check_gpu_concurrency(api_base_url: str) -> tuple[str, str] | None:
    """Return (experiment_id, workflow) if GPU is busy, else None.

    Parameters
    ----------
    api_base_url:
        Base URL of the HARS backend, e.g. ``http://localhost:8765``.

    Returns
    -------
    (experiment_id, workflow) or None
        None means the GPU is free (or the backend is unreachable — we fail open
        for availability, not safety, since indexing is non-destructive).
    """
    try:
        resp = requests.get(
            f"{api_base_url}/api/v1/experiments",
            params={"status": "running"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        experiments = (
            data
            if isinstance(data, list)
            else data.get("experiments", data.get("items", []))
        )
        for exp in experiments:
            workflow = str(exp.get("workflow_type", "")).lower()
            status = str(exp.get("status", "")).lower()
            if workflow in _GPU_EXCLUSIVE_WORKFLOWS and status == "running":
                return (str(exp.get("id", "unknown")), workflow)
        return None
    except Exception as exc:
        logger.warning(
            "GPU guard: backend unreachable (%s) — proceeding (fail-open).", exc
        )
        return None


def assert_gpu_free(api_base_url: str) -> None:
    """Raise GpuBusyError if a GPU-exclusive workflow is currently running.

    Call this at the start of any indexing operation.
    """
    result = check_gpu_concurrency(api_base_url)
    if result is not None:
        exp_id, workflow = result
        raise GpuBusyError(blocking_id=exp_id, workflow=workflow)
    logger.info("GPU guard: GPU is free, indexing may proceed.")
