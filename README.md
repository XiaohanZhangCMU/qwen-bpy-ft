# moonlake

Fine-tune Qwen2.5-Coder to generate Blender Python (`bpy`) scene programs from natural language prompts.

**Goal**: given a prompt like `"create a night market scene with local food stands"`, the model produces a runnable `bpy` script that builds the scene in Blender.

---

## Architecture

Three fully independent pipeline modules share one repo but have no cross-module Python imports:

```
data_collection/   →  data/raw/trajectories_<ts>.jsonl
                                ↓
training/          →  data/processed/moonlake_sft.jsonl  →  outputs/<checkpoint>/
                                                                       ↓
evaluation/        ←  configs/evaluation/<model>.yaml
```

Shared utilities (Blender runner, logging, config loader) live in `shared/`.

---

## Setup

### 1. Clone and install

```bash
# Python deps via uv (required on cluster)
uv pip install -e ".[dev]"

# LLaMA-Factory (training only)
uv pip install llamafactory
```

### 2. PyTorch for RTX 5090 / Blackwell (sm_120)

Standard PyTorch releases do not include sm_120 kernels. Install the nightly cu128 build:

```bash
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

> Skip this if you are on an older GPU (Ampere/Ada/Hopper). Standard `torch>=2.1` is fine.

### 3. vLLM

```bash
uv pip install vllm
```

### 4. Blender

```bash
export BLENDER_BIN=/path/to/blender          # Blender 4.x binary
echo 'export BLENDER_BIN=/path/to/blender' >> ~/.bashrc
```

### 5. Environment secrets

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY (required for data collection and GPT-4o eval)
```

### 6. Verify the environment

```bash
bash scripts/verify_env.sh
```

---

## Step-by-step workflow

### Step 1 — Collect training trajectories

```bash
bash scripts/collect.sh
# or with overrides:
python -m data_collection.pipeline \
  --config configs/data_collection/default.yaml \
  --target 150 \
  --model gpt-4o
```

Output: `data/raw/trajectories_<timestamp>.jsonl`

Each trajectory is a multi-turn conversation:
```
user prompt → bpy code → Blender execution → error feedback → repair → accepted scene
```

**Quality gates** (all four must pass):
1. Final script exits with code 0
2. Scene has ≥ 2 mesh objects
3. ≥ 3 total turns
4. ≥ 1 repair turn

**Loosening gates for faster bootstrapping** — edit `configs/data_collection/default.yaml`:
```yaml
quality:
  require_repair_turn: false   # allow trajectories with no repair
  min_turns: 1                 # accept shorter conversations
  min_objects: 1               # accept sparser scenes
```

---

### Step 2 — Prepare the training dataset

```bash
bash scripts/prepare.sh
```

Reads all `data/raw/*.jsonl`, filters to accepted trajectories, converts to LLaMA-Factory `sharegpt` format, and writes:
- `data/processed/moonlake_sft.jsonl`
- `data/processed/dataset_info.json`

---

### Step 3 — Train

```bash
# Qwen2.5-Coder-7B (default)
bash scripts/train.sh

# Qwen2.5-Coder-3B (faster iteration)
bash scripts/train.sh configs/training/qwen_sft_4b.yaml
```

Both configs use `CUDA_VISIBLE_DEVICES=4,5,6,7` and effective batch size 64 (1 sample × 16 grad accum × 4 GPUs).

Checkpoints saved to:
- `outputs/qwen2_5_coder_7b_lora/`
- `outputs/qwen2_5_coder_3b_lora/`

---

### Step 4 — Serve models with vLLM

Evaluation uses vLLM's OpenAI-compatible API. Start servers before running eval.

**3B server** (port 8000 — serves base model + fine-tuned adapter):
```bash
bash scripts/serve_3b.sh
```

**7B server** (port 8001 — serves fine-tuned adapter only):
```bash
bash scripts/serve_7b.sh
```

Both scripts use `CUDA_VISIBLE_DEVICES=4,5,6,7` and `--tensor-parallel-size 4`.

Each server registers LoRA adapters at startup:
| Server | Model ID | Adapter |
|--------|----------|---------|
| port 8000 | `Qwen/Qwen2.5-Coder-3B-Instruct` | base (no adapter) |
| port 8000 | `ft_qwen3b` | `outputs/qwen2_5_coder_3b_lora` |
| port 8001 | `ft_qwen7b` | `outputs/qwen2_5_coder_7b_lora` |

---

### Step 5 — Evaluate (one model at a time)

Results are saved to disk so you can run each model separately and compare later.

```bash
# Base Qwen-3B (needs serve_3b.sh running)
python -m evaluation.pipeline \
  --config configs/evaluation/base_model.yaml \
  --tag base_qwen3b

# Fine-tuned Qwen-3B (needs serve_3b.sh running)
python -m evaluation.pipeline \
  --config configs/evaluation/finetuned_3b.yaml \
  --tag ft_qwen3b

# Fine-tuned Qwen-7B (needs serve_7b.sh running)
python -m evaluation.pipeline \
  --config configs/evaluation/finetuned_7b.yaml \
  --tag ft_qwen7b

# GPT-4o oracle (no local server needed, uses OPENAI_API_KEY)
python -m evaluation.pipeline \
  --config configs/evaluation/openai.yaml \
  --tag openai
```

Results written to `data/eval/results_<tag>_<timestamp>.json`.

**Quick run on fewer prompts** (useful for smoke-testing):
```bash
python -m evaluation.pipeline --config configs/evaluation/finetuned_7b.yaml \
  --tag ft_qwen7b --num-prompts 5
```

**Run all four sequentially** (prints commands by default; add `--run` to execute):
```bash
bash scripts/eval_all.sh           # print the four commands
bash scripts/eval_all.sh --run     # run all four, then print table

# Limit to N prompts:
MOONLAKE_NUM_PROMPTS=5 bash scripts/eval_all.sh --run
```

---

### Step 6 — Compare results

```bash
bash scripts/compare.sh
```

Finds the latest result file for each tag and prints two tables:

**Quality table** (`pass@1`, `pass@3`, `pass@5`, `execution_success_rate`, `mean_n_objects`):
```
Model          pass@1   pass@3   pass@5   exec_success   mean_objs
Base Qwen-3B    0.30     0.52     0.63       0.31           3.1
FT Qwen-3B      0.55     0.74     0.82       0.56           5.2
FT Qwen-7B      0.68     0.85     0.91       0.69           6.8
GPT-4o          0.82     0.95     0.98       0.83           8.4
```

**Speed table** (`mean_generation_sec` per sample, lower is better):
```
Model          mean_gen_sec
Base Qwen-3B      1.8
FT Qwen-3B        1.9
FT Qwen-7B        4.2
GPT-4o           12.6
```

---

### Step 7 — Visualize a scene (optional)

Generate a scene from a prompt and render 4 views (perspective, front, side, top):

```bash
bash scripts/render.sh "a cozy bedroom with a bed, wardrobe, and reading lamp"

# Use a specific backend:
bash scripts/render.sh "a night market" --backend openai
bash scripts/render.sh "a night market" --backend vllm --checkpoint ft_qwen7b

# Render an existing script:
bash scripts/render.sh --script path/to/scene.py
```

Output saved to `data/renders/<timestamp>/`: four PNG views + `scene.blend` + `scene.py`.

---

## Eval metrics reference

| Metric | Description |
|--------|-------------|
| `pass@1` | Probability a single generation succeeds (unbiased estimator, Chen et al.) |
| `pass@3`, `pass@5` | Probability ≥1 of k generations succeeds (unbiased estimator) |
| `execution_success_rate` | Raw fraction of all generated scripts that exit with code 0 |
| `mean_n_objects` | Average mesh object count in passing scenes |
| `mean_generation_sec` | Average wall-clock time per sample generation |

**`pass@1` vs `execution_success_rate`**: `pass@1` is the unbiased estimator computed from `num_samples_per_prompt` (default 5) samples per prompt, then averaged across prompts (macro). `execution_success_rate` is the raw micro-average: `total_passing_scripts / total_scripts`. They are close but differ because `pass@1` weights each prompt equally regardless of how many samples it generated.

---

## File layout

```
qwen-bpy-ft/
├── shared/                     # Shared utilities (no domain logic)
│   ├── blender_runner.py       # Headless Blender execution + manifest injection
│   ├── config.py               # YAML loader with env-var overrides
│   └── logging_utils.py        # JSON structured logging
├── data_collection/            # Stage 1: collect multi-turn trajectories
│   ├── pipeline.py             # Main loop
│   ├── generator.py            # LLM API wrapper
│   ├── quality_gate.py         # Four quality gates
│   ├── scene_verifier.py       # Parse scene manifest
│   ├── prompt_templates.py     # System prompt + turn formatters + scene seeds
│   └── schemas.py              # Pydantic models
├── training/                   # Stage 2: SFT via LLaMA-Factory
│   ├── prepare_dataset.py      # JSONL → sharegpt format
│   └── train.py                # Wrapper around llamafactory-cli
├── evaluation/                 # Stage 3: eval + compare + render
│   ├── pipeline.py             # Per-model eval loop
│   ├── infer.py                # HF / vLLM / OpenAI backends
│   ├── metrics.py              # pass@k, aggregation
│   ├── compare.py              # Multi-model comparison table
│   ├── render.py               # 4-view Blender rendering
│   └── schemas.py              # Result pydantic models
├── configs/
│   ├── data_collection/default.yaml
│   ├── training/
│   │   ├── qwen_sft.yaml       # 7B LoRA config
│   │   └── qwen_sft_4b.yaml    # 3B LoRA config
│   └── evaluation/
│       ├── base_model.yaml     # Base Qwen-3B via vLLM port 8000
│       ├── finetuned_3b.yaml   # FT Qwen-3B via vLLM port 8000
│       ├── finetuned_7b.yaml   # FT Qwen-7B via vLLM port 8001
│       └── openai.yaml         # GPT-4o oracle
├── scripts/
│   ├── verify_env.sh
│   ├── collect.sh
│   ├── prepare.sh
│   ├── train.sh
│   ├── serve_3b.sh             # vLLM server: 3B base + ft_qwen3b on port 8000
│   ├── serve_7b.sh             # vLLM server: ft_qwen7b on port 8001
│   ├── eval.sh                 # Single-model eval
│   ├── eval_all.sh             # All four models sequentially
│   ├── compare.sh              # Print comparison table
│   └── render.sh               # Scene generation + 4-view render
├── data/
│   ├── raw/                    # Collected trajectories (JSONL)
│   ├── processed/              # LLaMA-Factory training format
│   └── eval/                   # Prompts + per-model results + comparisons
├── outputs/                    # LoRA checkpoints
├── .env.example
└── pyproject.toml
```

---

## Trajectory schema (JSONL)

```jsonc
{
  "id": "<uuid>",
  "seed": "a cozy bedroom with a bed, wardrobe, and reading lamp",
  "model_id": "gpt-4o",
  "created_at": "2026-03-30T14:00:00Z",
  "quality": {
    "passed": true,
    "failed_gates": [],
    "n_turns": 5,
    "n_repair_turns": 1,
    "n_objects": 8
  },
  "turns": [
    { "role": "user",      "content": "Create a Blender 3D scene: ..." },
    { "role": "assistant", "content": "```python\nimport bpy\n...```",
      "execution": { "exit_code": 1, "stderr": "NameError: ...", "elapsed_sec": 2.1 } },
    { "role": "tool",      "content": "Blender exited with code 1. Stderr: ..." },
    { "role": "user",      "content": "Please fix the error above." },
    { "role": "assistant", "content": "```python\nimport bpy\n...```",
      "execution": { "exit_code": 0, "stderr": "", "elapsed_sec": 1.8 } }
  ]
}
```

---

## Success targets

| Metric | Target |
|--------|--------|
| Accepted trajectories | ≥ 100 |
| Median turns per trajectory | ≥ 3 |
| Trajectories with ≥1 repair | ≥ 80% |
| Final headless exec success | ≥ 95% |
| FT model `pass@1` vs base | higher |
| FT model `execution_success_rate` vs base | higher |
