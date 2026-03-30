# moonlake

Fine-tune a VLM to generate Blender Python (`bpy`) scene programs from natural language prompts.

**End goal**: a user can prompt the model with e.g. `"create a night market scene with local food stands"` and the model produces a runnable `bpy` script that builds the scene in Blender.

---

## Architecture

Three fully independent pipeline modules share one repo but have no cross-module Python imports:

```
data_collection/   →  data/raw/*.jsonl
                                ↓
training/          →  data/processed/moonlake_sft.jsonl  →  outputs/<checkpoint>/
                                                                       ↓
evaluation/        ←  configs/evaluation/default.yaml  (checkpoint path)
```

The only shared code lives in `shared/` (Blender runner, logging, config loader).

---

## Quickstart

### 0. Prerequisites (remote cluster)

```bash
# Blender (headless, 3.x or 4.x)
export BLENDER_BIN=/path/to/blender

# Python deps
pip install -e .

# LLaMA-Factory (training only)
pip install llamafactory

# Copy and fill in secrets
cp .env.example .env
```

### 1. Check your environment

```bash
bash scripts/verify_env.sh
```

### 2. Collect trajectories

```bash
bash scripts/collect.sh                          # uses configs/data_collection/default.yaml
# or override inline:
python -m data_collection.pipeline --config configs/data_collection/default.yaml --target 150
```

Writes JSONL to `data/raw/trajectories_<timestamp>.jsonl`.

Each trajectory is a multi-turn conversation:
- user request → assistant bpy code → Blender execution → error feedback → repair → accepted scene

### 3. Prepare training dataset

```bash
bash scripts/prepare.sh
```

Filters to accepted trajectories, deduplicates, converts to LLaMA-Factory `sharegpt` format,
and writes `data/processed/moonlake_sft.jsonl` + `dataset_info.json`.

### 4. Train

```bash
bash scripts/train.sh                            # 7B default
bash scripts/train.sh configs/training/qwen_sft_4b.yaml   # 3B fast iteration
```

Checkpoint saved to `outputs/qwen2_5_coder_7b_lora/`.

### 5. Evaluate

```bash
bash scripts/eval.sh                             # uses configs/evaluation/default.yaml
# or specify a checkpoint directly:
python -m evaluation.pipeline --checkpoint outputs/qwen2_5_coder_7b_lora
```

Results written to `data/eval/results_<timestamp>.json`.

---

## Module reference

| Module | Entry point | Config |
|--------|------------|--------|
| `data_collection` | `python -m data_collection.pipeline` | `configs/data_collection/default.yaml` |
| `training` prepare | `python -m training.prepare_dataset` | (CLI args) |
| `training` train | `python -m training.train` | `configs/training/qwen_sft.yaml` |
| `evaluation` | `python -m evaluation.pipeline` | `configs/evaluation/default.yaml` |

---

## Trajectory schema (JSONL)

```jsonc
{
  "id": "<uuid>",
  "seed": "a cozy bedroom with a bed, wardrobe, and reading lamp",
  "model_id": "gpt-4o",
  "created_at": "2026-03-30T14:00:00Z",
  "quality": { "passed": true, "failed_gates": [], "n_turns": 5, "n_repair_turns": 1, "n_objects": 8 },
  "turns": [
    { "role": "user",      "content": "Create a Blender 3D scene: ..." },
    { "role": "assistant", "content": "```python\nimport bpy\n...```",
      "execution": { "exit_code": 1, "stderr": "...", "elapsed_sec": 2.1 } },
    { "role": "tool",      "content": "Blender exited with code 1. Stderr: ..." },
    { "role": "user",      "content": "Please fix the error..." },
    { "role": "assistant", "content": "```python\nimport bpy\n...```",
      "execution": { "exit_code": 0, "stderr": "", "elapsed_sec": 1.8 } }
  ]
}
```

**Quality gates** (all must pass for a trajectory to be accepted):
1. Final script exits with code 0 (no Python exceptions)
2. Scene has ≥ 2 mesh objects
3. ≥ 3 total turns
4. ≥ 1 repair/feedback turn

---

## Eval metrics

| Metric | Description |
|--------|-------------|
| `pass@1` | Fraction of prompts where the single-shot script succeeds |
| `pass@k` (k=3,5) | Unbiased estimator: probability ≥1 of k samples passes |
| `execution_success_rate` | Raw fraction of scripts that exit 0 across all samples |
| `mean_n_objects` | Average scene object count in passing runs |

---

## Success targets

| Gate | Target |
|------|--------|
| `accepted_trajectories` | ≥ 100 |
| `median_turns` | ≥ 3 |
| `repair_trajectory_rate` | ≥ 80% |
| `final_headless_exec_success` | ≥ 95% |
| Eval `pass@1` vs base model | higher |
| Eval `execution_success_rate` | higher |
