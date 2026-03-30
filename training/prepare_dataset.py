"""
Convert raw trajectory JSONL files into LLaMA-Factory sharegpt format.

Usage:
    python -m training.prepare_dataset \
        --input  data/raw/ \
        --output data/processed/moonlake_sft.jsonl \
        --dataset-info data/processed/dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from data_collection.schemas import Trajectory
from data_collection.prompt_templates import SYSTEM_PROMPT
from shared.logging_utils import get_logger

logger = get_logger(__name__)

DATASET_NAME = "moonlake_sft"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def trajectory_to_sharegpt(trajectory: Trajectory) -> Optional[dict]:
    """
    Convert one Trajectory to LLaMA-Factory sharegpt format.

    Mapping:
      - role "user"      → from "human"
      - role "assistant" → from "gpt"
      - role "tool"      → folded into the NEXT "user" turn
        (prepended as context, since sharegpt SFT has no native tool role)

    Only turns up to and including the final passing assistant turn are kept.
    Returns None if the trajectory has no assistant turns.
    """
    turns = trajectory.turns
    if not turns:
        return None

    conversations: list[dict] = [
        {"from": "system", "value": SYSTEM_PROMPT}
    ]

    pending_tool_context: list[str] = []

    for i, turn in enumerate(turns):
        if turn.role == "tool":
            pending_tool_context.append(turn.content)
        elif turn.role == "user":
            content = turn.content
            if pending_tool_context:
                prefix = "\n\n".join(pending_tool_context)
                content = f"{prefix}\n\n{content}"
                pending_tool_context = []
            conversations.append({"from": "human", "value": content})
        elif turn.role == "assistant":
            if pending_tool_context:
                # Tool context with no following user turn — attach to assistant
                # by inserting a synthetic human turn
                prefix = "\n\n".join(pending_tool_context)
                conversations.append({"from": "human", "value": prefix})
                pending_tool_context = []
            conversations.append({"from": "gpt", "value": turn.content})

    # Must end on a "gpt" turn
    if not conversations or conversations[-1]["from"] != "gpt":
        return None

    # Must have at least one human/gpt exchange
    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not (has_human and has_gpt):
        return None

    return {"conversations": conversations}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_by_seed(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Keep at most one trajectory per exact seed string."""
    seen: set[str] = set()
    result: list[Trajectory] = []
    for t in trajectories:
        if t.seed not in seen:
            seen.add(t.seed)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Dataset info patcher
# ---------------------------------------------------------------------------

def patch_dataset_info(dataset_info_path: Path, output_jsonl: Path) -> None:
    """
    Create or update a LLaMA-Factory dataset_info.json entry for our dataset.
    """
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            info = json.load(f)
    else:
        info = {}

    info[DATASET_NAME] = {
        "file_name": str(output_jsonl.name),
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }

    with open(dataset_info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info("dataset_info.json updated", extra={"path": str(dataset_info_path)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_trajectories(input_path: Path) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    paths = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    for p in paths:
        with open(p) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = Trajectory.model_validate_json(line)
                    trajectories.append(traj)
                except Exception as e:
                    logger.warning(
                        "Skipping malformed trajectory",
                        extra={"file": str(p), "line": lineno, "error": str(e)},
                    )
    return trajectories


def prepare(
    input_path: Path,
    output_path: Path,
    dataset_info_path: Optional[Path] = None,
    require_passed: bool = True,
) -> int:
    """
    Load, filter, convert, and write the dataset.

    Returns the number of examples written.
    """
    logger.info("Loading trajectories", extra={"input": str(input_path)})
    all_trajectories = load_trajectories(input_path)
    logger.info("Loaded", extra={"total": len(all_trajectories)})

    # Filter: only accepted trajectories
    if require_passed:
        accepted = [t for t in all_trajectories if t.quality and t.quality.passed]
        logger.info(
            "Filtered to accepted",
            extra={"accepted": len(accepted), "rejected": len(all_trajectories) - len(accepted)},
        )
    else:
        accepted = all_trajectories

    # Dedup
    deduped = dedup_by_seed(accepted)
    logger.info("After dedup", extra={"count": len(deduped)})

    # Convert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as f:
        for traj in deduped:
            example = trajectory_to_sharegpt(traj)
            if example is None:
                logger.warning("Skipping trajectory (failed conversion)", extra={"id": traj.id})
                continue
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Dataset written", extra={"examples": written, "output": str(output_path)})

    # Update dataset_info.json
    if dataset_info_path:
        patch_dataset_info(dataset_info_path, output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for LLaMA-Factory SFT")
    parser.add_argument("--input", default="data/raw", help="Raw trajectory JSONL dir or file")
    parser.add_argument(
        "--output", default="data/processed/moonlake_sft.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--dataset-info",
        default="data/processed/dataset_info.json",
        help="Path to LLaMA-Factory dataset_info.json to update",
    )
    parser.add_argument(
        "--all", action="store_true", help="Include rejected trajectories (default: accepted only)"
    )
    args = parser.parse_args()

    n = prepare(
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_info_path=Path(args.dataset_info),
        require_passed=not args.all,
    )
    print(f"Wrote {n} examples to {args.output}")
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
