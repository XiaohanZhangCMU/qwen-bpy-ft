#!/usr/bin/env python3
"""Extract bpy scripts from an eval results JSON for manual inspection in Blender.

Usage:
    python scripts/extract_scripts.py results_ft_qwen7b_20260330T225757.json
    python scripts/extract_scripts.py results.json --out-dir /tmp/blender_scripts
    python scripts/extract_scripts.py results.json --only-passing
    python scripts/extract_scripts.py results.json --only-failing
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_file", type=Path, help="Path to results JSON file")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <results_stem>_scripts/)")
    parser.add_argument("--only-passing", action="store_true", help="Only extract runs with exit_code=0")
    parser.add_argument("--only-failing", action="store_true", help="Only extract runs with exit_code!=0")
    args = parser.parse_args()

    data = json.loads(args.results_file.read_text())
    out_dir = args.out_dir or args.results_file.parent / f"{args.results_file.stem}_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for pr in data.get("prompt_results", []):
        prompt_id = pr["prompt_id"]
        prompt_text = pr.get("prompt", "")
        for run in pr.get("runs", []):
            idx = run.get("sample_idx", 0)
            exit_code = run.get("exit_code", -1)
            script = run.get("script", "")
            if not script:
                continue
            if args.only_passing and exit_code != 0:
                continue
            if args.only_failing and exit_code == 0:
                continue

            status = "pass" if exit_code == 0 else f"fail{exit_code}"
            fname = out_dir / f"{prompt_id}_s{idx}_{status}.py"
            header = (
                f"# prompt_id : {prompt_id}\n"
                f"# sample_idx: {idx}\n"
                f"# exit_code : {exit_code}\n"
                f"# prompt    : {prompt_text}\n"
                f"# {'=' * 72}\n\n"
            )
            fname.write_text(header + script)
            print(f"  {'OK' if exit_code == 0 else 'FAIL':4s}  {fname.name}")
            extracted += 1

    print(f"\n{extracted} script(s) written to: {out_dir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
