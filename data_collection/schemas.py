"""
Pydantic schemas for data collection artifacts.

The canonical on-disk format is JSONL where each line is a Trajectory.model_dump_json().
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from shared.blender_runner import ExecutionResult


# ---------------------------------------------------------------------------
# Per-turn
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    # Only set on assistant turns that were executed
    execution: Optional[ExecutionResult] = None


# ---------------------------------------------------------------------------
# Quality gate result
# ---------------------------------------------------------------------------

class QualityResult(BaseModel):
    passed: bool
    failed_gates: list[str] = Field(default_factory=list)
    n_turns: int = 0
    n_repair_turns: int = 0
    n_objects: int = 0


# ---------------------------------------------------------------------------
# Full trajectory (one training example)
# ---------------------------------------------------------------------------

class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seed: str                      # original scene description prompt
    model_id: str                  # which LLM generated the code turns
    turns: list[Turn] = Field(default_factory=list)
    quality: Optional[QualityResult] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Convenience accessors
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def tool_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "tool"]

    def last_assistant_code(self) -> Optional[str]:
        from shared.blender_runner import extract_python_block
        for t in reversed(self.turns):
            if t.role == "assistant":
                return extract_python_block(t.content)
        return None
