"""
Compare evaluation results across multiple model runs and print a metrics table.

Usage:
    python -m evaluation.compare \
        --results data/eval/results_base_<ts>.json \
                  data/eval/results_finetuned_<ts>.json \
                  data/eval/results_openai_<ts>.json \
        --labels  "Base Qwen-3B" "Fine-tuned Qwen-3B" "GPT-4o"
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _print_section(
    title: str,
    metrics: list[tuple[str, str, str, bool]],  # (display, key, fmt, higher_is_better)
    rows: list[dict],
    labels: list[str],
    col_w: int,
    metric_w: int,
) -> None:
    print(f"\n{title}")
    header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    for display_name, key, fmt, higher_is_better in metrics:
        values = [r.get(key, float("nan")) for r in rows]
        valid = [v for v in values if v == v]
        best = max(valid) if higher_is_better else min(valid) if valid else float("nan")
        row = f"{display_name:<{metric_w}}"
        for v in values:
            cell = fmt.format(v) if v == v else "N/A"
            if v == v and v == best:
                cell = cell + "*"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print("* = best in row")

    # Delta vs baseline
    if len(rows) >= 2:
        print(f"\nΔ vs baseline ({labels[0]}):")
        delta_header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels[1:])
        print(delta_header)
        print("-" * len(delta_header))
        for display_name, key, fmt, _ in metrics:
            base_val = rows[0].get(key, float("nan"))
            row = f"{display_name:<{metric_w}}"
            for r in rows[1:]:
                v = r.get(key, float("nan"))
                if v != v or base_val != base_val:
                    row += f"{'N/A':>{col_w}}"
                else:
                    delta = v - base_val
                    sign = "+" if delta >= 0 else ""
                    cell = f"{sign}{delta:.1%}" if "%" in fmt.format(0.0) else f"{sign}{delta:.2f}s" if "s" in fmt else f"{sign}{delta:.1f}"
                    row += f"{cell:>{col_w}}"
            print(row)


def print_table(rows: list[dict], labels: list[str]) -> None:
    col_w = max(len(l) for l in labels) + 2
    metric_w = 22

    quality_metrics = [
        ("pass@1",             "macro_pass_at_1",        "{:.1%}", True),
        ("pass@3",             "macro_pass_at_3",        "{:.1%}", True),
        ("pass@5",             "macro_pass_at_5",        "{:.1%}", True),
        ("exec success rate",  "execution_success_rate", "{:.1%}", True),
        ("mean objects/scene", "mean_n_objects",         "{:.1f}", True),
    ]

    speed_metrics = [
        ("avg generation time", "mean_generation_sec", "{:.2f}s", False),  # lower is better
    ]

    _print_section("=== Quality metrics ===", quality_metrics, rows, labels, col_w, metric_w)
    _print_section("=== Speed metrics ===",   speed_metrics,   rows, labels, col_w, metric_w)
    print()


def find_latest(tag: str, results_dir: str = "data/eval") -> Path:
    """Return the most recently modified results file whose name contains *tag*."""
    matches = sorted(
        Path(results_dir).glob(f"results_{tag}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No result file found for tag '{tag}' in {results_dir}/\n"
            f"Run: python -m evaluation.pipeline --config <cfg> --tag {tag}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eval results across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # By explicit file paths:
          python -m evaluation.compare \\
            --results data/eval/results_base_qwen3b_*.json \\
                      data/eval/results_ft_qwen3b_*.json \\
                      data/eval/results_ft_qwen7b_*.json \\
                      data/eval/results_openai_*.json \\
            --labels "Base Qwen-3B" "FT Qwen-3B" "FT Qwen-7B" "GPT-4o"

          # By tag (automatically finds the latest run for each):
          python -m evaluation.compare \\
            --tags base_qwen3b ft_qwen3b ft_qwen7b openai \\
            --labels "Base Qwen-3B" "FT Qwen-3B" "FT Qwen-7B" "GPT-4o"
        """),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--results", nargs="+",
        help="Explicit paths to result JSON files",
    )
    src.add_argument(
        "--tags", nargs="+",
        help="Tags to look up (finds latest results_<tag>_*.json in data/eval/)",
    )
    parser.add_argument("--labels", nargs="+", help="Display names (same order as results/tags)")
    parser.add_argument("--results-dir", default="data/eval", help="Directory to search when using --tags")
    parser.add_argument("--save", help="Optional path to save the comparison as JSON")
    args = parser.parse_args()

    if args.tags:
        paths = [str(find_latest(t, args.results_dir)) for t in args.tags]
        default_labels = args.tags
    else:
        paths = args.results
        default_labels = [Path(p).stem for p in paths]

    if args.labels and len(args.labels) != len(paths):
        parser.error("--labels must have the same count as --results/--tags")

    labels = args.labels or default_labels
    rows = [load_summary(p) for p in paths]

    # Print which files are being compared
    print("\nLoading results:")
    for label, p in zip(labels, paths):
        print(f"  {label:20s}  {p}")

    print_table(rows, labels)

    if args.save:
        comparison = {
            label: {
                "macro_pass_at_1":        r.get("macro_pass_at_1"),
                "macro_pass_at_3":        r.get("macro_pass_at_3"),
                "macro_pass_at_5":        r.get("macro_pass_at_5"),
                "execution_success_rate": r.get("execution_success_rate"),
                "mean_n_objects":         r.get("mean_n_objects"),
                "mean_generation_sec":    r.get("mean_generation_sec"),
            }
            for label, r in zip(labels, rows)
        }
        with open(args.save, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
