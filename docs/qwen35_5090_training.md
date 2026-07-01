# Qwen3.5 2B OPSD LoRA Training on 4x RTX 5090

This document describes the Qwen3.5 2B OPSD LoRA launch script:

```bash
scripts/run_opsd_qwen35_2b_5090.sh
```

It is configured for 4 GPUs, DeepSpeed ZeRO-2, LoRA training, student non-thinking rollout, and teacher thinking scoring.

## Quick Start

Run a 1-step smoke test first:

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

WANDB_MODE=offline \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
MAX_STEPS=1 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

Then run a short online W&B test:

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

export WANDB_API_KEY="your_wandb_api_key"

OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
RUN_CONFIG=qwen35_2b_opsd_lora_50step \
MAX_STEPS=50 \
SAVE_STEPS=50 \
LOGGING_STEPS=5 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

If that is stable, start a 1-epoch training run:

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

export WANDB_API_KEY="your_wandb_api_key"

OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
RUN_CONFIG=qwen35_2b_opsd_lora_1epoch \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS=500 \
LOGGING_STEPS=10 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

## W&B

The launch script supports online W&B through environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `WANDB_API_KEY` | unset | If set, the script runs `wandb login --relogin` and defaults to online logging. Do not commit this value. |
| `WANDB_MODE` | `online` when `WANDB_API_KEY` is set, otherwise `offline` | Use `online` for normal tracking, `offline` for local-only logs, or `disabled` to disable W&B. |
| `WANDB_PROJECT` | `OPSD` | W&B project name. |
| `WANDB_ENTITY` | unset | Optional W&B entity/team/user. Set only if your account needs it. |

Safer interactive key entry:

```bash
read -s WANDB_API_KEY
export WANDB_API_KEY
```

One-line key passing also works, but may be saved in shell history:

```bash
WANDB_API_KEY="your_wandb_api_key" \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
RUN_CONFIG=qwen35_2b_opsd_lora \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS=500 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

## Important Training Parameters

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | `/Users/hyh/Desktop/Qwen3.5_2B` | Local path to the base Qwen3.5 model. On the server, use `/DATA_A/models/Qwen3.5-2B`. |
| `OPSD_DATASET` | `siyanzhao/Openthoughts_math_30k_opsd` | HF dataset name or local `.json/.jsonl/.csv` file. Your current file uses `input/output` JSONL. |
| `OUTPUT_DIR` | `./outputs/opsd_qwen35_2b_5090` | Output root. The script appends `RUN_CONFIG`. |
| `RUN_CONFIG` | `qwen35_2b_5090_lora_nonthink_topk256` | Run/checkpoint subdirectory and W&B run prefix. |
| `NUM_PROCESSES` | `4` | Number of GPUs/processes. For 4x 5090, keep `4`. |
| `PER_DEVICE_BATCH_SIZE` | `1` | Micro-batch size per GPU. |
| `GRAD_ACCUM_STEPS` | `8` | Gradient accumulation. Effective batch size is `PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES`, default `32`. |
| `NUM_TRAIN_EPOCHS` | `30` | Epoch count when `MAX_STEPS` is not set. For this dataset, start with `1`. |
| `MAX_STEPS` | unset | Hard step limit. Useful for smoke tests and short runs. Overrides epoch-based stopping. |
| `SAVE_STEPS` | `25` | Checkpoint save interval. For long runs, use `500` or larger to avoid too many checkpoints. |
| `LOGGING_STEPS` | `2` | Logging interval. |
| `LEARNING_RATE` | `5e-6` | LoRA learning rate. |
| `LORA_R` | `64` | LoRA rank. |
| `LORA_ALPHA` | `128` | LoRA alpha. |
| `MAX_LENGTH` | `8192` | Max prompt/context length for collator tokenization. |
| `MAX_COMPLETION_LENGTH` | `1024` | Student rollout max new tokens. |
| `TEMPERATURE` | `1.0` | Student generation temperature. |
| `TOP_P` | `1.0` | Student generation top-p. |
| `TOP_K` | `20` | Student generation top-k. |
| `PRESENCE_PENALTY` | `2.0` | Generation presence penalty. |
| `TOP_K_LOSS` | `256` | Restrict token-level distillation loss to teacher top-k tokens. |
| `JSD_TOKEN_CLIP` | `1e-6` | Per-token JSD clipping for stability. |

## Thinking Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `STUDENT_THINKING` | `False` | Student rollout is non-thinking. |
| `TEACHER_THINKING` | `True` | Teacher scoring prompt uses Qwen thinking mode. |
| `CLOSE_TEACHER_THINKING_BEFORE_SCORING` | `True` | Adds a teacher-only hidden thinking closure before scoring student tokens, reducing format mismatch risk. |
| `REAPPLY_CHAT_TEMPLATE_TO_INPUT` | `True` | Re-parses `input` ChatML and applies the current Qwen3.5 chat template. |

This is the intended setup: strong teacher in thinking mode guides a non-thinking student, while the student output remains final-answer style.

## Dataset Format

Your current SFT-style JSONL format is supported:

```json
{"input": "<|im_start|>user\n...\n<|im_end|><|im_start|>assistant\n", "output": "..."}
```

The code reads `input/output` by default. If a dataset uses different column names, override:

```bash
INPUT_FIELD=prompt OUTPUT_FIELD=response bash scripts/run_opsd_qwen35_2b_5090.sh
```

## Recommended Run Progression

1. `MAX_STEPS=1`: verify environment, model loading, data loading, generation, backward pass.
2. `MAX_STEPS=50`: verify W&B online logging, speed, loss curves, checkpoint writing.
3. `NUM_TRAIN_EPOCHS=1`: first real training run.
4. Longer runs only after checking output quality and checkpoint size.

With 106,919 examples and default effective batch size 32, one epoch is about 3,342 optimizer steps.

## Checkpoints and Inference

Checkpoints are saved under:

```bash
/DATA_B/hyh/opsd_outputs/<RUN_CONFIG>/
```

Generate with base model plus a LoRA checkpoint:

```bash
python scripts/generate_qwen35_torch.py \
  --model /DATA_A/models/Qwen3.5-2B \
  --adapter /DATA_B/hyh/opsd_outputs/qwen35_2b_opsd_lora_1epoch/checkpoint-500 \
  --prompt "Summarize these points into one sentence: ..."
```

Warnings about the Qwen3.5 fast path falling back to the torch implementation are not blocking. The model has already trained successfully in the smoke test with that fallback.
