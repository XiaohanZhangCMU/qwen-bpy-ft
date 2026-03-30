"""
Compute pass@k and other evaluation metrics.

Reference: Chen et al. "Evaluating Large Language Models Trained on Code" (HumanEval).
The unbiased pass@k estimator avoids sampling bias.
"""

from __future__ import annotations

import math
from typing import Optional

from evaluation.schemas import EvalSummary, PromptResult, SingleRunResult


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute the unbiased pass@k estimate.

    Args:
        n: Total number of samples generated per problem.
        c: Number of samples that pass (exit_code == 0 and scene non-empty).
        k: k in pass@k.

    Returns:
        Probability that at least one of k samples passes.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Per-prompt aggregation
# ---------------------------------------------------------------------------

def aggregate_prompt(
    prompt_id: str,
    prompt: str,
    runs: list[SingleRunResult],
    ks: tuple[int, ...] = (1, 3, 5),
) -> PromptResult:
    n = len(runs)
    c = sum(1 for r in runs if r.exit_code == 0 and r.n_mesh >= 1)
    mean_objs = sum(r.n_objects for r in runs) / max(n, 1)
    mean_elapsed = sum(r.elapsed_sec for r in runs) / max(n, 1)

    result = PromptResult(
        prompt_id=prompt_id,
        prompt=prompt,
        runs=runs,
        mean_n_objects=round(mean_objs, 2),
        mean_elapsed_sec=round(mean_elapsed, 3),
    )
    if 1 in ks:
        result.pass_at_1 = pass_at_k(n, c, 1)
    if 3 in ks:
        result.pass_at_3 = pass_at_k(n, c, min(3, n))
    if 5 in ks:
        result.pass_at_5 = pass_at_k(n, c, min(5, n))
    return result


# ---------------------------------------------------------------------------
# Summary across all prompts
# ---------------------------------------------------------------------------

def compute_summary(
    model_id: str,
    checkpoint_dir: str,
    prompt_results: list[PromptResult],
) -> EvalSummary:
    n = len(prompt_results)
    if n == 0:
        return EvalSummary(model_id=model_id, checkpoint_dir=checkpoint_dir)

    macro_p1 = sum(r.pass_at_1 for r in prompt_results) / n
    macro_p3 = sum(r.pass_at_3 for r in prompt_results) / n
    macro_p5 = sum(r.pass_at_5 for r in prompt_results) / n
    mean_objs = sum(r.mean_n_objects for r in prompt_results) / n

    all_runs = [run for pr in prompt_results for run in pr.runs]
    exec_success = sum(1 for r in all_runs if r.exit_code == 0) / max(len(all_runs), 1)

    return EvalSummary(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        n_prompts=n,
        macro_pass_at_1=round(macro_p1, 4),
        macro_pass_at_3=round(macro_p3, 4),
        macro_pass_at_5=round(macro_p5, 4),
        mean_n_objects=round(mean_objs, 2),
        execution_success_rate=round(exec_success, 4),
        prompt_results=prompt_results,
    )
