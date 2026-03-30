"""
Evaluation pipeline: runs the fine-tuned model on held-out prompts,
executes the generated scripts in headless Blender, and computes metrics.

Usage:
    python -m evaluation.pipeline --config configs/evaluation/default.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from evaluation.infer import build_inferencer
from evaluation.metrics import aggregate_prompt, compute_summary
from evaluation.schemas import SingleRunResult
from shared.blender_runner import extract_python_block, run_blender_script
from shared.config import load_config
from shared.logging_utils import get_logger
from data_collection.scene_verifier import read_manifest

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    checkpoint_dir: str = "outputs/qwen_lora"
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    inference_backend: str = "hf"       # "hf" or "vllm"
    temperature: float = 0.0
    max_new_tokens: int = 2048


class EvaluationConfig(BaseModel):
    prompts_file: str = "data/eval/prompts.jsonl"
    output_dir: str = "data/eval"
    pass_at_k: list[int] = Field(default_factory=lambda: [1, 3, 5])
    num_samples_per_prompt: int = 5
    max_repair_turns: int = 0    # 0 = no self-repair during eval (single-shot)


class BlenderConfig(BaseModel):
    binary: Optional[str] = None
    timeout_sec: int = 60
    headless: bool = True


class EvalPipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_prompt(
    prompt_id: str,
    prompt: str,
    inferencer,
    cfg: EvalPipelineConfig,
) -> list[SingleRunResult]:
    """Generate *num_samples* scripts for *prompt* and execute each."""
    import time

    n = cfg.evaluation.num_samples_per_prompt

    # Time the full generate() call, then split evenly per sample
    t0 = time.monotonic()
    samples = inferencer.generate(prompt, n=n)
    generation_sec_per_sample = round((time.monotonic() - t0) / max(len(samples), 1), 3)

    results: list[SingleRunResult] = []
    for idx, sample in enumerate(samples):
        code = extract_python_block(sample)
        if code is None:
            results.append(SingleRunResult(
                prompt_id=prompt_id,
                sample_idx=idx,
                exit_code=-3,
                stderr="No Python code block found in model output.",
                generation_sec=generation_sec_per_sample,
                script=sample,
            ))
            continue

        exec_result = run_blender_script(
            script_content=code,
            blender_bin=cfg.blender.binary,
            timeout_sec=cfg.blender.timeout_sec,
            inject_manifest=True,
        )
        scene_info = read_manifest(exec_result.manifest_path)

        results.append(SingleRunResult(
            prompt_id=prompt_id,
            sample_idx=idx,
            exit_code=exec_result.exit_code,
            stdout=exec_result.stdout[-500:],
            stderr=exec_result.stderr[-1000:],
            elapsed_sec=exec_result.elapsed_sec,
            generation_sec=generation_sec_per_sample,
            n_objects=scene_info.n_objects,
            n_mesh=scene_info.n_mesh,
            script=code,
        ))

    return results


def run_eval(cfg: EvalPipelineConfig, num_prompts: Optional[int] = None, tag: Optional[str] = None) -> Path:
    """Main evaluation loop. Returns path to the summary JSON."""
    # Load prompts
    prompts_path = Path(cfg.evaluation.prompts_file)
    prompts: list[dict] = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if num_prompts:
        prompts = prompts[:num_prompts]
    logger.info("Loaded eval prompts", extra={"n": len(prompts)})

    # Build inferencer
    inferencer = build_inferencer(
        backend=cfg.model.inference_backend,
        model_name_or_path=cfg.model.model_name_or_path,
        checkpoint_dir=cfg.model.checkpoint_dir,
        temperature=cfg.model.temperature,
        max_new_tokens=cfg.model.max_new_tokens,
    )

    # Evaluate
    from evaluation.schemas import PromptResult
    prompt_results: list[PromptResult] = []
    ks = tuple(cfg.evaluation.pass_at_k)

    for item in tqdm(prompts, desc="Evaluating prompts"):
        pid = item.get("id", "unknown")
        prompt_text = item.get("prompt", "")
        logger.info("Evaluating prompt", extra={"id": pid})
        runs = evaluate_prompt(pid, prompt_text, inferencer, cfg)
        pr = aggregate_prompt(pid, prompt_text, runs, ks=ks)
        prompt_results.append(pr)
        logger.info(
            "Prompt result",
            extra={"id": pid, "pass@1": pr.pass_at_1, "n_runs": len(runs)},
        )

    # Summary
    summary = compute_summary(
        model_id=cfg.model.model_name_or_path,
        checkpoint_dir=cfg.model.checkpoint_dir,
        prompt_results=prompt_results,
    )

    # Write output
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    out_path = output_dir / f"results{tag_part}_{ts}.json"
    with open(out_path, "w") as f:
        f.write(summary.model_dump_json(indent=2))

    logger.info(
        "Evaluation complete",
        extra={
            "pass@1": summary.macro_pass_at_1,
            "pass@5": summary.macro_pass_at_5,
            "exec_success": summary.execution_success_rate,
            "output": str(out_path),
        },
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned bpy model")
    parser.add_argument(
        "--config", default="configs/evaluation/default.yaml", help="Evaluation config YAML"
    )
    parser.add_argument("--backend", choices=["hf", "vllm", "openai"], help="Override inference backend")
    parser.add_argument("--checkpoint", help="Override checkpoint_dir")
    parser.add_argument("--num-prompts", type=int, help="Only evaluate the first N prompts (quick test)")
    parser.add_argument("--tag", help="Label embedded in the output filename, e.g. 'base_qwen3b'")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_config(args.config, EvalPipelineConfig)
    if args.backend:
        cfg.model.inference_backend = args.backend
    if args.checkpoint:
        cfg.model.checkpoint_dir = args.checkpoint

    out = run_eval(cfg, num_prompts=args.num_prompts, tag=args.tag)
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
