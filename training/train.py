"""
Training entry point: validates the config and shells out to llamafactory-cli.

Usage:
    python -m training.train --config configs/training/qwen_sft.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def run_training(config_path: str | Path) -> int:
    """
    Call ``llamafactory-cli train <config_path>`` as a subprocess.

    Returns the process exit code.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Config not found", extra={"path": str(config_path)})
        return 1

    cmd = ["llamafactory-cli", "train", str(config_path)]
    logger.info("Launching LLaMA-Factory", extra={"cmd": " ".join(cmd)})

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        logger.error("Training failed", extra={"exit_code": proc.returncode})
    else:
        logger.info("Training complete", extra={"exit_code": proc.returncode})

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT training via LLaMA-Factory")
    parser.add_argument(
        "--config",
        default="configs/training/qwen_sft.yaml",
        help="Path to LLaMA-Factory training YAML",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    exit_code = run_training(args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
