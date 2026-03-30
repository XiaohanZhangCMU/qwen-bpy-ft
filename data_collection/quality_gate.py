"""
Quality gate: checks the four acceptance criteria for a collected trajectory.
"""

from __future__ import annotations

from data_collection.schemas import QualityResult, Trajectory
from data_collection.scene_verifier import SceneInfo
from shared.logging_utils import get_logger

logger = get_logger(__name__)

# Configurable thresholds (can be overridden from config)
MIN_TURNS = 3
MIN_OBJECTS = 2
REQUIRE_REPAIR_TURN = True
REQUIRE_ZERO_EXIT = True


def check(
    trajectory: Trajectory,
    scene_info: SceneInfo,
    min_turns: int = MIN_TURNS,
    min_objects: int = MIN_OBJECTS,
    require_repair_turn: bool = REQUIRE_REPAIR_TURN,
) -> QualityResult:
    """
    Evaluate *trajectory* against the acceptance gates.

    Gates (all must pass):
    1. Final assistant turn's script exited with code 0.
    2. Scene has at least *min_objects* mesh objects.
    3. Trajectory has at least *min_turns* turns total.
    4. At least one "tool" turn exists (i.e. a repair/feedback round happened).

    Returns:
        QualityResult with passed=True only if all gates pass.
    """
    failed: list[str] = []

    # Gate 1: exit code
    assistant_turns = trajectory.assistant_turns()
    if not assistant_turns:
        failed.append("no_assistant_turns")
    else:
        last_exec = assistant_turns[-1].execution
        if last_exec is None or last_exec.exit_code != 0:
            ec = last_exec.exit_code if last_exec else "N/A"
            failed.append(f"final_exit_code_nonzero:{ec}")

    # Gate 2: scene richness
    if scene_info.n_mesh < min_objects:
        failed.append(f"insufficient_mesh_objects:{scene_info.n_mesh}<{min_objects}")

    # Gate 3: turn count
    n_turns = len(trajectory.turns)
    if n_turns < min_turns:
        failed.append(f"too_few_turns:{n_turns}<{min_turns}")

    # Gate 4: at least one repair turn
    n_repair = len(trajectory.tool_turns())
    if require_repair_turn and n_repair == 0:
        failed.append("no_repair_turn")

    passed = len(failed) == 0
    result = QualityResult(
        passed=passed,
        failed_gates=failed,
        n_turns=n_turns,
        n_repair_turns=n_repair,
        n_objects=scene_info.n_objects,
    )
    logger.info(
        "Quality check",
        extra={"passed": passed, "failed_gates": failed, "trajectory_id": trajectory.id},
    )
    return result
