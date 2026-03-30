"""
Data collection pipeline: orchestrates the full trajectory collection loop.

Entry point:  python -m data_collection.pipeline --config configs/data_collection/default.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from data_collection.conversation import Conversation
from data_collection.executor import Executor
from data_collection.generator import Generator
from data_collection import quality_gate
from data_collection.prompt_templates import (
    SCENE_SEEDS,
    format_initial_user_turn,
    format_layout_feedback_turn,
    format_repair_user_turn,
    format_scene_check_turn,
)
from data_collection.schemas import Trajectory
from data_collection.scene_verifier import read_manifest
from shared.config import load_config
from shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class GenerationConfig(BaseModel):
    model_id: str = "gpt-4o"
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.8
    max_tokens: int = 2048
    max_repair_turns: int = 4


class CollectionConfig(BaseModel):
    target_trajectories: int = 150
    max_attempts_per_seed: int = 8
    seeds_file: Optional[str] = None
    output_dir: str = "data/raw"
    output_prefix: str = "trajectories"


class BlenderConfig(BaseModel):
    binary: Optional[str] = None   # falls back to BLENDER_BIN env var
    timeout_sec: int = 60
    headless: bool = True


class QualityConfig(BaseModel):
    min_turns: int = 3
    min_objects: int = 2
    require_repair_turn: bool = True
    require_zero_exit_code: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    fmt: str = "json"


class DataCollectionConfig(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def collect_one_trajectory(
    seed: str,
    generator: Generator,
    executor: Executor,
    cfg: DataCollectionConfig,
) -> Optional[Trajectory]:
    """
    Attempt to collect one accepted trajectory for *seed*.

    Returns the Trajectory if it passes quality gates, else None.
    """
    conv = Conversation(seed=seed, model_id=generator.model_id)

    # Turn 1: initial user request
    conv.add_user(format_initial_user_turn(seed))

    scene_info = None
    for attempt in range(1, cfg.generation.max_repair_turns + 2):
        # --- Generate code ---
        try:
            reply = generator.complete(conv.messages)
        except Exception as e:
            logger.error("Generator error", extra={"error": str(e), "seed": seed})
            return None

        # --- Execute ---
        _code, exec_result = executor.run(reply)
        conv.add_assistant(reply, execution=exec_result)

        # --- Parse scene manifest ---
        scene_info = read_manifest(exec_result.manifest_path)

        if exec_result.exit_code == 0:
            # Check scene richness — may trigger a content repair turn
            if scene_info.n_mesh < cfg.quality.min_objects and attempt <= cfg.generation.max_repair_turns:
                feedback = format_scene_check_turn(scene_info.n_objects, scene_info.n_mesh)
                conv.add_tool(feedback)
                conv.add_user(feedback)
                continue
            if scene_info.issues and attempt <= cfg.generation.max_repair_turns:
                feedback = format_layout_feedback_turn(scene_info.issues)
                conv.add_tool(feedback)
                conv.add_user(feedback)
                continue
            # Success — exit the repair loop
            break
        else:
            # Execution failed — give the model the error and ask for a repair
            if attempt > cfg.generation.max_repair_turns:
                break
            tool_msg = format_repair_user_turn(exec_result.stderr, exec_result.exit_code, attempt)
            conv.add_tool(tool_msg)
            conv.add_user(tool_msg)

    # --- Quality gate ---
    trajectory = conv.to_trajectory()
    qr = quality_gate.check(
        trajectory,
        scene_info=scene_info or type("_S", (), {"n_mesh": 0, "n_objects": 0})(),  # type: ignore[arg-type]
        min_turns=cfg.quality.min_turns,
        min_objects=cfg.quality.min_objects,
        require_repair_turn=cfg.quality.require_repair_turn,
    )
    trajectory.quality = qr
    return trajectory


def run_collection(cfg: DataCollectionConfig) -> Path:
    """
    Main collection loop.  Writes accepted trajectories to a JSONL file.

    Returns the path to the output file.
    """
    import os

    # -- Seeds --
    if cfg.collection.seeds_file:
        seeds_path = Path(cfg.collection.seeds_file)
        seeds = seeds_path.read_text().splitlines()
    else:
        seeds = list(SCENE_SEEDS)

    random.shuffle(seeds)
    seed_pool = seeds * ((cfg.collection.max_attempts_per_seed * cfg.collection.target_trajectories // len(seeds)) + 2)

    # -- Output file --
    output_dir = Path(cfg.collection.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = output_dir / f"{cfg.collection.output_prefix}_{ts}.jsonl"

    # -- Components --
    import os as _os
    api_key = _os.environ.get(cfg.generation.api_key_env, "")
    generator = Generator(
        model_id=cfg.generation.model_id,
        api_base=cfg.generation.api_base,
        api_key=api_key,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
    )
    executor = Executor(
        blender_bin=cfg.blender.binary,
        timeout_sec=cfg.blender.timeout_sec,
    )

    accepted = 0
    attempted = 0
    seed_iter = iter(seed_pool)

    with open(output_path, "w") as out_f:
        with tqdm(total=cfg.collection.target_trajectories, desc="Collecting trajectories") as pbar:
            while accepted < cfg.collection.target_trajectories:
                try:
                    seed = next(seed_iter)
                except StopIteration:
                    logger.warning("Ran out of seeds before reaching target")
                    break

                attempted += 1
                logger.info("Attempting trajectory", extra={"seed": seed[:60], "accepted": accepted})

                trajectory = collect_one_trajectory(seed, generator, executor, cfg)

                if trajectory and trajectory.quality and trajectory.quality.passed:
                    out_f.write(trajectory.model_dump_json() + "\n")
                    out_f.flush()
                    accepted += 1
                    pbar.update(1)
                    logger.info(
                        "Trajectory accepted",
                        extra={"id": trajectory.id, "accepted": accepted, "attempted": attempted},
                    )
                else:
                    gates = trajectory.quality.failed_gates if trajectory and trajectory.quality else ["unknown"]
                    logger.info("Trajectory rejected", extra={"failed_gates": gates})

    logger.info(
        "Collection complete",
        extra={"accepted": accepted, "attempted": attempted, "output": str(output_path)},
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bpy trajectory data")
    parser.add_argument(
        "--config",
        default="configs/data_collection/default.yaml",
        help="Path to data_collection config YAML",
    )
    parser.add_argument("--target", type=int, help="Override target_trajectories")
    parser.add_argument("--model", help="Override generation.model_id")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_config(args.config, DataCollectionConfig)

    if args.target:
        cfg.collection.target_trajectories = args.target
    if args.model:
        cfg.generation.model_id = args.model

    logger.info("Starting collection", extra={"config": cfg.model_dump()})
    output_path = run_collection(cfg)
    print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
