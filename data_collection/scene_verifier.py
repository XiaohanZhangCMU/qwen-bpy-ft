"""
Parse the JSON scene manifest written by the injected snippet in blender_runner.py.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel

from shared.logging_utils import get_logger

logger = get_logger(__name__)


class SceneInfo(BaseModel):
    n_objects: int = 0
    n_mesh: int = 0
    n_light: int = 0
    n_camera: int = 0
    object_names: list[str] = []
    issues: list[str] = []


def read_manifest(manifest_path: Optional[str]) -> SceneInfo:
    """
    Load the scene manifest JSON written by blender_runner's injected snippet.

    Returns an empty SceneInfo (with an issue recorded) if the manifest
    doesn't exist or can't be parsed.
    """
    if not manifest_path or not os.path.exists(manifest_path):
        return SceneInfo(issues=["scene manifest not found"])

    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read scene manifest", extra={"error": str(e)})
        return SceneInfo(issues=[f"manifest parse error: {e}"])
    finally:
        # Clean up the temp manifest file
        try:
            os.unlink(manifest_path)
        except OSError:
            pass

    objects = data.get("objects", [])
    info = SceneInfo(
        n_objects=data.get("n_objects", len(objects)),
        n_mesh=data.get("n_mesh", sum(1 for o in objects if o.get("type") == "MESH")),
        n_light=data.get("n_light", sum(1 for o in objects if o.get("type") == "LIGHT")),
        n_camera=data.get("n_camera", sum(1 for o in objects if o.get("type") == "CAMERA")),
        object_names=[o.get("name", "") for o in objects],
    )

    # Lightweight QC checks
    if info.n_objects == 0:
        info.issues.append("scene is empty (0 objects)")
    if info.n_mesh < 2:
        info.issues.append(f"too few mesh objects ({info.n_mesh}); expected >= 2")
    if info.n_light == 0:
        info.issues.append("no lights in scene")
    if info.n_camera == 0:
        info.issues.append("no camera in scene")

    return info
