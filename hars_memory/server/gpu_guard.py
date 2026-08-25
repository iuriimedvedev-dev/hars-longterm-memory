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


class GpuGuardUnavailableError(RuntimeError):
    """Raised when the HARS backend cannot be reached to confirm GPU state.

    Fail-CLOSED: if we cannot ask the backend whether a GPU-exclusive workflow
    is running, indexing must not proceed — otherwise it can start on top of a
    live training run and compete for the GPU. The operator who knows the GPU
    is idle can override via ``allow_unreachable_backend=True``
    (``index.py --allow-gpu-guard-unreachable`` /
    ``HARS_MEMORY_GPU_GUARD_ALLOW_UNREACHABLE=1``).
    """

    def __init__(self, api_base_url: str, reason: str) -> None:
        super().__init__(
            f"GPU guard: backend unreachable at {api_base_url} ({reason}). "
            "Cannot confirm the GPU is free — refusing to start indexing. "
            "If you know the GPU is idle, override with "
            "--allow-gpu-guard-unreachable (or HARS_MEMORY_GPU_GUARD_ALLOW_UNREACHABLE=1)."
        )
        self.api_base_url = api_base_url
        self.reason = reason


def check_gpu_concurrency(
    api_base_url: str,
    *,
    allow_unreachable_backend: bool = False,
) -> tuple[str, str] | None:
    """Return (experiment_id, workflow) if GPU is busy, else None.

    Parameters
    ----------
    api_base_url:
        Base URL of the HARS backend, e.g. ``http://localhost:8765``.
    allow_unreachable_backend:
        Explicit operator override. When False (default), an unreachable/
        erroring backend raises ``GpuGuardUnavailableError`` (fail CLOSED —
        we cannot confirm the GPU is free, so we must not proceed). When
        True, the same condition is treated as "GPU is free" and logged.

    Returns
    -------
    (experiment_id, workflow) or None
        None means the GPU is confirmed free.

    Raises
    ------
    GpuGuardUnavailableError
        If the backend is unreachable/erroring and *allow_unreachable_backend*
        is False.
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
        if allow_unreachable_backend:
            logger.warning(
                "GPU guard: backend unreachable (%s) — proceeding because "
                "allow_unreachable_backend=True was explicitly set.", exc,
            )
            return None
        raise GpuGuardUnavailableError(api_base_url, str(exc)) from exc


def assert_gpu_free(
    api_base_url: str,
    *,
    allow_unreachable_backend: bool = False,
) -> None:
    """Raise GpuBusyError / GpuGuardUnavailableError unless the GPU is confirmed free.

    Call this at the start of any indexing operation.
    """
    result = check_gpu_concurrency(
        api_base_url, allow_unreachable_backend=allow_unreachable_backend
    )
    if result is not None:
        exp_id, workflow = result
        raise GpuBusyError(blocking_id=exp_id, workflow=workflow)
    logger.info("GPU guard: GPU is free, indexing may proceed.")
