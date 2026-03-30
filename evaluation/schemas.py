"""Pydantic schemas for evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class SingleRunResult(BaseModel):
    """Result of running one generated script in headless Blender."""
    prompt_id: str
    sample_idx: int
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0       # Blender execution time
    generation_sec: float = 0.0    # model inference time for this sample
    n_objects: int = 0
    n_mesh: int = 0
    script: Optional[str] = None


class PromptResult(BaseModel):
    """Aggregated results for one eval prompt across all samples."""
    prompt_id: str
    prompt: str
    runs: list[SingleRunResult] = Field(default_factory=list)
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    pass_at_5: float = 0.0
    mean_n_objects: float = 0.0
    mean_elapsed_sec: float = 0.0
    mean_generation_sec: float = 0.0  # avg inference time per sample


class EvalSummary(BaseModel):
    """Top-level summary across all eval prompts."""
    model_id: str
    checkpoint_dir: str
    n_prompts: int = 0
    macro_pass_at_1: float = 0.0
    macro_pass_at_3: float = 0.0
    macro_pass_at_5: float = 0.0
    mean_n_objects: float = 0.0
    execution_success_rate: float = 0.0
    mean_generation_sec: float = 0.0   # avg inference time across all prompts
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_results: list[PromptResult] = Field(default_factory=list)
