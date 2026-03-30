"""
Manages the multi-turn conversation state for a single trajectory.
"""

from __future__ import annotations

from data_collection.prompt_templates import SYSTEM_PROMPT
from data_collection.schemas import Turn, Trajectory
from shared.blender_runner import ExecutionResult


class Conversation:
    """
    Stateful multi-turn conversation for one scene-generation trajectory.

    Tracks the message history in OpenAI API format (for passing to Generator)
    and the structured Turn list (for saving to JSONL).
    """

    def __init__(self, seed: str, model_id: str) -> None:
        self.seed = seed
        self.model_id = model_id
        self._turns: list[Turn] = []
        # The API message list always starts with the system prompt
        self._messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ------------------------------------------------------------------
    # Add turns
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, execution: ExecutionResult | None = None) -> None:
        self._turns.append(Turn(role="assistant", content=content, execution=execution))
        self._messages.append({"role": "assistant", "content": content})

    def add_tool(self, content: str) -> None:
        """
        Add a tool/runtime-feedback turn.

        The feedback is visible to the model as a "user" message
        (most OpenAI-compatible APIs don't have a "tool" role for plain text),
        but we tag it as "tool" in our internal Turn schema for QC/filtering.
        """
        self._turns.append(Turn(role="tool", content=content))
        # Surface as user in the API message list so the model sees it
        self._messages.append({"role": "user", "content": content})

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        """The full message history in OpenAI API format."""
        return self._messages

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def n_assistant_turns(self) -> int:
        return sum(1 for t in self._turns if t.role == "assistant")

    def last_execution(self) -> ExecutionResult | None:
        for t in reversed(self._turns):
            if t.role == "assistant" and t.execution is not None:
                return t.execution
        return None

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------

    def to_trajectory(self) -> Trajectory:
        return Trajectory(
            seed=self.seed,
            model_id=self.model_id,
            turns=self._turns,
        )
