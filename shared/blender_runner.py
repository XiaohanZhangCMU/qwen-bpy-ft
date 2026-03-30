"""
Canonical wrapper for running a bpy script in headless Blender.

Both data_collection and evaluation import from here — nowhere else.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    elapsed_sec: float
    script_path: Optional[str] = None   # temp file used, if any
    manifest_path: Optional[str] = None  # scene JSON manifest, if written


# ---------------------------------------------------------------------------
# Scene-manifest injection snippet
# ---------------------------------------------------------------------------
# Appended to every script so we can inspect the scene without re-running it.
_MANIFEST_SNIPPET = textwrap.dedent("""
# --- moonlake scene manifest (auto-injected) ---
import json as _json, os as _os
_manifest_path = _os.environ.get("MOONLAKE_MANIFEST_PATH", "")
if _manifest_path:
    try:
        import bpy as _bpy
        _objects = [
            {"name": o.name, "type": o.type, "location": list(o.location)}
            for o in _bpy.context.scene.objects
        ]
        _manifest = {
            "objects": _objects,
            "n_objects": len(_objects),
            "n_mesh": sum(1 for o in _objects if o["type"] == "MESH"),
            "n_light": sum(1 for o in _objects if o["type"] == "LIGHT"),
            "n_camera": sum(1 for o in _objects if o["type"] == "CAMERA"),
        }
        with open(_manifest_path, "w") as _f:
            _json.dump(_manifest, _f, indent=2)
    except Exception as _e:
        pass  # never let manifest writing kill the script
""")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_blender_script(
    script_content: str,
    blender_bin: Optional[str] = None,
    timeout_sec: int = 60,
    inject_manifest: bool = True,
    extra_args: Optional[list[str]] = None,
) -> ExecutionResult:
    """
    Execute *script_content* in headless Blender.

    Args:
        script_content:  Full Python source to run inside Blender.
        blender_bin:     Path to the Blender binary.  Falls back to
                         BLENDER_BIN env var, then ``blender`` on PATH.
        timeout_sec:     Subprocess timeout.
        inject_manifest: Append the scene-manifest dump snippet.
        extra_args:      Extra CLI args appended after ``--python <file>``.

    Returns:
        ExecutionResult with exit_code, stdout, stderr, elapsed_sec,
        and manifest_path if inject_manifest=True.
    """
    bin_path = blender_bin or os.environ.get("BLENDER_BIN", "blender")

    manifest_path: Optional[str] = None
    env = os.environ.copy()

    if inject_manifest:
        manifest_fd, manifest_path = tempfile.mkstemp(suffix=".json", prefix="moonlake_manifest_")
        os.close(manifest_fd)
        env["MOONLAKE_MANIFEST_PATH"] = manifest_path
        script_content = script_content + _MANIFEST_SNIPPET

    # Write the script to a temp file
    script_fd, script_path = tempfile.mkstemp(suffix=".py", prefix="moonlake_script_")
    try:
        with os.fdopen(script_fd, "w") as f:
            f.write(script_content)

        cmd = [
            bin_path,
            "--background",
            "--factory-startup",
            "--python-exit-code", "1",
            "--python", script_path,
        ]
        if extra_args:
            cmd.extend(extra_args)

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
            elapsed = time.monotonic() - t0
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_sec=round(elapsed, 3),
                script_path=script_path,
                manifest_path=manifest_path,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Blender process timed out after {timeout_sec}s",
                elapsed_sec=round(elapsed, 3),
                script_path=script_path,
                manifest_path=manifest_path,
            )
        except FileNotFoundError:
            return ExecutionResult(
                exit_code=-2,
                stdout="",
                stderr=f"Blender binary not found: {bin_path}",
                elapsed_sec=0.0,
                script_path=script_path,
                manifest_path=manifest_path,
            )
    finally:
        # Clean up script temp file; manifest is cleaned up by caller
        try:
            os.unlink(script_path)
        except OSError:
            pass


def extract_python_block(text: str) -> Optional[str]:
    """
    Extract the first ```python ... ``` fenced code block from *text*.
    Returns None if no block is found.
    """
    import re
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if text looks like raw Python (starts with import/bpy)
    stripped = text.strip()
    if stripped.startswith("import ") or stripped.startswith("import bpy"):
        return stripped
    return None
