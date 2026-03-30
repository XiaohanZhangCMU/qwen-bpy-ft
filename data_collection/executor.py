"""
Executor: extracts the bpy script from an assistant message and runs it in Blender.
"""

from __future__ import annotations

from typing import Optional

from shared.blender_runner import ExecutionResult, extract_python_block, run_blender_script
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class Executor:
    """
    Wraps shared.blender_runner for use inside the data-collection loop.

    Args:
        blender_bin:  Path to the Blender binary.
        timeout_sec:  Per-run timeout.
    """

    def __init__(self, blender_bin: Optional[str] = None, timeout_sec: int = 60) -> None:
        self.blender_bin = blender_bin
        self.timeout_sec = timeout_sec

    def run(self, assistant_message: str) -> tuple[Optional[str], ExecutionResult]:
        """
        Extract a Python code block from *assistant_message* and execute it.

        Returns:
            (code, result) where *code* is the extracted Python string
            (or None if no code block was found) and *result* is the
            ExecutionResult (exit_code=-3 if no code found).
        """
        code = extract_python_block(assistant_message)
        if code is None:
            logger.warning("No Python code block found in assistant message")
            return None, ExecutionResult(
                exit_code=-3,
                stdout="",
                stderr="No Python code block found in the assistant message.",
                elapsed_sec=0.0,
            )

        logger.debug("Executing bpy script", extra={"code_len": len(code)})
        result = run_blender_script(
            script_content=code,
            blender_bin=self.blender_bin,
            timeout_sec=self.timeout_sec,
            inject_manifest=True,
        )
        logger.info(
            "Execution finished",
            extra={"exit_code": result.exit_code, "elapsed_sec": result.elapsed_sec},
        )
        return code, result
