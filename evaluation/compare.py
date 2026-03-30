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
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_table(rows: list[dict], labels: list[str]) -> None:
    metrics = [
        ("pass@1",               "macro_pass_at_1",       "{:.1%}"),
        ("pass@3",               "macro_pass_at_3",       "{:.1%}"),
        ("pass@5",               "macro_pass_at_5",       "{:.1%}"),
        ("exec success rate",    "execution_success_rate","{:.1%}"),
        ("mean objects/scene",   "mean_n_objects",        "{:.1f}"),
    ]

    col_w = max(len(l) for l in labels) + 2
    metric_w = 20

    # Header
    header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print()
    print(header)
    print("-" * len(header))

    for display_name, key, fmt in metrics:
        row = f"{display_name:<{metric_w}}"
        values = [r.get(key, float("nan")) for r in rows]
        best = max(v for v in values if v == v)  # nan-safe max
        for v in values:
            cell = fmt.format(v) if v == v else "N/A"
            # Bold the best value with an asterisk
            if v == v and v == best:
                cell = cell + "*"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print("* = best in row")
    print()

    # Delta columns: improvement of each non-baseline over the first entry
    if len(rows) >= 2:
        print("Δ vs baseline (first column):")
        delta_header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels[1:])
        print(delta_header)
        print("-" * len(delta_header))
        for display_name, key, fmt in metrics:
            row = f"{display_name:<{metric_w}}"
            base_val = rows[0].get(key, float("nan"))
            for r in rows[1:]:
                v = r.get(key, float("nan"))
                if v != v or base_val != base_val:
                    row += f"{'N/A':>{col_w}}"
                else:
                    delta = v - base_val
                    sign = "+" if delta >= 0 else ""
                    # Use same format but show as delta
                    if "%" in fmt.format(0.0):
                        cell = f"{sign}{delta:.1%}"
                    else:
                        cell = f"{sign}{delta:.1f}"
                    row += f"{cell:>{col_w}}"
            print(row)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval results across models")
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Paths to result JSON files (in order: base, finetuned, openai, ...)"
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="Display names for each result file (same order)"
    )
    parser.add_argument(
        "--save", help="Optional path to save the comparison as JSON"
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.results):
        parser.error("--labels must have the same count as --results")

    labels = args.labels or [Path(p).stem for p in args.results]
    rows = [load_summary(p) for p in args.results]

    print_table(rows, labels)

    if args.save:
        comparison = {
            label: {
                "macro_pass_at_1":      r.get("macro_pass_at_1"),
                "macro_pass_at_3":      r.get("macro_pass_at_3"),
                "macro_pass_at_5":      r.get("macro_pass_at_5"),
                "execution_success_rate": r.get("execution_success_rate"),
                "mean_n_objects":       r.get("mean_n_objects"),
            }
            for label, r in zip(labels, rows)
        }
        with open(args.save, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
