"""
Generate a bpy script from a prompt and render it to PNG images via headless Blender.

Usage:
    # From a text prompt (uses the fine-tuned model):
    python -m evaluation.render \
        --prompt "a cozy bedroom with a bed, wardrobe, and reading lamp" \
        --checkpoint outputs/qwen2_5_coder_3b_lora \
        --out-dir renders/bedroom

    # From an already-saved script file:
    python -m evaluation.render --script my_scene.py --out-dir renders/bedroom

    # Using GPT-4o instead:
    python -m evaluation.render \
        --prompt "a night market with food stalls" \
        --backend openai --out-dir renders/market
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from shared.blender_runner import extract_python_block
from shared.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Render injection snippet
# Appended to the user script to set up camera, lighting, and multi-view render.
# The user script may already have a camera/lights — this snippet only adds
# what is missing, then renders 4 views (front, side, top, perspective).
# ---------------------------------------------------------------------------

RENDER_SNIPPET = textwrap.dedent("""
# --- moonlake render (auto-injected) ---
import bpy as _bpy, math as _math, os as _os

_out_dir = _os.environ.get("MOONLAKE_RENDER_DIR", "/tmp/moonlake_render")
_os.makedirs(_out_dir, exist_ok=True)

# Ensure there is exactly one camera
_cams = [o for o in _bpy.context.scene.objects if o.type == "CAMERA"]
if not _cams:
    _bpy.ops.object.camera_add(location=(7, -7, 5))
    _cam = _bpy.context.active_object
    _cam.rotation_euler = (_math.radians(60), 0, _math.radians(45))
    _bpy.context.scene.camera = _cam
else:
    _bpy.context.scene.camera = _cams[0]

# Ensure there is at least one light
_lights = [o for o in _bpy.context.scene.objects if o.type == "LIGHT"]
if not _lights:
    _bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
    _bpy.context.active_object.data.energy = 3.0

# Render settings
_scene = _bpy.context.scene
_scene.render.engine = "CYCLES"
_scene.cycles.samples = 64          # low sample count — fast preview
_scene.render.resolution_x = 800
_scene.render.resolution_y = 600
_scene.render.image_settings.file_format = "PNG"
_scene.render.film_transparent = False

# Compute scene bounding centre for camera targeting
def _scene_centre():
    coords = [
        v.co @ o.matrix_world
        for o in _bpy.context.scene.objects
        if o.type == "MESH"
        for v in o.data.vertices
    ]
    if not coords:
        import mathutils
        return mathutils.Vector((0, 0, 0))
    from mathutils import Vector
    xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
    return Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))

_centre = _scene_centre()
_radius = 10.0   # camera orbit radius

# Four standard views: perspective, front, side, top
_views = [
    ("perspective", (_math.radians(60), 0, _math.radians(45))),
    ("front",       (_math.radians(75), 0, _math.radians(0))),
    ("side",        (_math.radians(75), 0, _math.radians(90))),
    ("top",         (_math.radians(0),  0, _math.radians(0))),
]

import mathutils as _mu
for _name, _rot in _views:
    # Position camera on orbit sphere, pointing at scene centre
    _r = _mu.Euler(_rot, "XYZ").to_matrix().to_4x4()
    _offset = (_r @ _mu.Vector((0, -_radius, _radius * 0.6))).to_3d()
    _bpy.context.scene.camera.location = _centre + _offset
    _bpy.context.scene.camera.rotation_euler = _rot

    # Point camera at centre
    _dir = _centre - _bpy.context.scene.camera.location
    _bpy.context.scene.camera.rotation_euler = _dir.to_track_quat("-Z", "Y").to_euler()

    _scene.render.filepath = _os.path.join(_out_dir, f"{_name}.png")
    _bpy.ops.render.render(write_still=True)
    print(f"Rendered: {_scene.render.filepath}")

print(f"All renders saved to: {_out_dir}")
""")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def render_prompt(
    prompt: Optional[str] = None,
    script_path: Optional[str] = None,
    out_dir: str = "renders/scene",
    blender_bin: Optional[str] = None,
    timeout_sec: int = 300,
    checkpoint_dir: Optional[str] = None,
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    backend: str = "hf",
    save_blend: bool = True,
) -> Path:
    """
    Generate (or load) a bpy script, inject the render snippet, execute it.

    Returns the output directory path containing the PNG renders and
    optionally a .blend file.
    """
    # --- Get the script ---
    if script_path:
        code = Path(script_path).read_text()
        logger.info("Loaded script from file", extra={"path": script_path})
    elif prompt:
        logger.info("Generating bpy script", extra={"prompt": prompt[:80]})
        from evaluation.infer import build_inferencer
        inferencer = build_inferencer(
            backend=backend,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            temperature=0.0,
            max_new_tokens=2048,
        )
        reply = inferencer.generate(prompt, n=1)[0]
        code = extract_python_block(reply)
        if code is None:
            print("ERROR: model did not produce a Python code block.", file=sys.stderr)
            print("Raw reply:\n", reply, file=sys.stderr)
            sys.exit(1)
    else:
        print("ERROR: provide --prompt or --script", file=sys.stderr)
        sys.exit(1)

    # --- Inject render + optional .blend save ---
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Save the raw script for reference
    script_out = out_dir_path / "scene.py"
    script_out.write_text(code)
    logger.info("Script saved", extra={"path": str(script_out)})

    blend_snippet = ""
    if save_blend:
        blend_path = str(out_dir_path / "scene.blend")
        blend_snippet = f'\n_bpy.ops.wm.save_as_mainfile(filepath="{blend_path}")\nprint("Blend saved:", "{blend_path}")\n'

    full_script = code + RENDER_SNIPPET + blend_snippet

    # Write combined script to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="moonlake_render_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_script)

        bin_path = blender_bin or os.environ.get("BLENDER_BIN", "blender")
        env = {**os.environ, "MOONLAKE_RENDER_DIR": str(out_dir_path)}

        cmd = [
            bin_path, "--background", "--factory-startup",
            "--python-exit-code", "1",
            "--python", tmp_path,
        ]
        logger.info("Running Blender render", extra={"out_dir": str(out_dir_path)})
        result = subprocess.run(cmd, env=env, timeout=timeout_sec, text=True,
                                capture_output=False)  # let output stream to terminal
        if result.returncode != 0:
            print(f"\nBlender exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    finally:
        os.unlink(tmp_path)

    renders = sorted(out_dir_path.glob("*.png"))
    print(f"\nRenders ({len(renders)} views): {out_dir_path}/")
    for r in renders:
        print(f"  {r.name}")
    if save_blend:
        print(f"  scene.blend  (open in Blender GUI)")
    print(f"  scene.py     (raw bpy script)")

    return out_dir_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Render a bpy scene from a prompt or script")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt",  help="Natural language scene description")
    src.add_argument("--script",  help="Path to an existing .py bpy script")

    parser.add_argument("--out-dir",    default="renders/scene", help="Output directory for PNGs and .blend")
    parser.add_argument("--checkpoint", default="outputs/qwen2_5_coder_3b_lora", help="LoRA checkpoint dir")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Base model HF id")
    parser.add_argument("--backend",    default="hf", choices=["hf", "vllm", "openai"])
    parser.add_argument("--no-blend",   action="store_true", help="Skip saving .blend file")
    parser.add_argument("--timeout",    type=int, default=300, help="Blender timeout in seconds")
    args = parser.parse_args()

    render_prompt(
        prompt=args.prompt,
        script_path=args.script,
        out_dir=args.out_dir,
        timeout_sec=args.timeout,
        checkpoint_dir=args.checkpoint,
        model_name_or_path=args.model,
        backend=args.backend,
        save_blend=not args.no_blend,
    )


if __name__ == "__main__":
    main()
