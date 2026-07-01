#!/usr/bin/env bash
set -euo pipefail

# Override these on the 5090 box if the paths differ:
#   MODEL_DIR=/data/models/Qwen3.5_2B OUTPUT_DIR=/data/opsd bash scripts/run_opsd_qwen35_2b_5090.sh
MODEL_DIR="${MODEL_DIR:-/Users/hyh/Desktop/Qwen3.5_2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/opsd_qwen35_2b_5090}"
OPSD_DATASET="${OPSD_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
OPSD_DATASET_SPLIT="${OPSD_DATASET_SPLIT:-train}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_5090_zero2.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
STUDENT_THINKING="${STUDENT_THINKING:-False}"
TEACHER_THINKING="${TEACHER_THINKING:-True}"
CLOSE_TEACHER_THINKING_BEFORE_SCORING="${CLOSE_TEACHER_THINKING_BEFORE_SCORING:-True}"
REAPPLY_CHAT_TEMPLATE_TO_INPUT="${REAPPLY_CHAT_TEMPLATE_TO_INPUT:-True}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_ENTITY
fi

if [[ "$WANDB_MODE" == "offline" || "$WANDB_MODE" == "disabled" ]]; then
    echo "[INFO] Running with WANDB_MODE=$WANDB_MODE"
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is not set; wandb may prompt or fail."
elif command -v wandb >/dev/null 2>&1; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null
    echo "[INFO] W&B online logging enabled: project=$WANDB_PROJECT"
else
    echo "[WARN] wandb CLI not found; Python wandb will use WANDB_API_KEY if the package is installed."
fi

EXTRA_TRAINING_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    EXTRA_TRAINING_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    EXTRA_TRAINING_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train.py \
    --model_name_or_path "$MODEL_DIR" \
    --model_loader image_text_to_text \
    --opsd_dataset "$OPSD_DATASET" \
    --opsd_dataset_split "$OPSD_DATASET_SPLIT" \
    --input_field "${INPUT_FIELD:-input}" \
    --output_field "${OUTPUT_FIELD:-output}" \
    --problem_field "${PROBLEM_FIELD:-problem}" \
    --solution_field "${SOLUTION_FIELD:-solution}" \
    --learning_rate "${LEARNING_RATE:-5e-6}" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --run_config "${RUN_CONFIG:-qwen35_2b_5090_lora_nonthink_topk256}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-30}" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_steps "${SAVE_STEPS:-25}" \
    --logging_steps "${LOGGING_STEPS:-2}" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --beta 0 \
    --use_peft \
    --lora_r "${LORA_R:-64}" \
    --lora_alpha "${LORA_ALPHA:-128}" \
    --lora_target_modules \
        q_proj k_proj v_proj o_proj \
        in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
        gate_proj up_proj down_proj \
    --temperature "${TEMPERATURE:-1.0}" \
    --top_p "${TOP_P:-1.0}" \
    --top_k "${TOP_K:-20}" \
    --presence_penalty "${PRESENCE_PENALTY:-2.0}" \
    --lmbda 1 \
    --fixed_teacher \
    --student_thinking "$STUDENT_THINKING" \
    --teacher_thinking "$TEACHER_THINKING" \
    --close_teacher_thinking_before_scoring "$CLOSE_TEACHER_THINKING_BEFORE_SCORING" \
    --reapply_chat_template_to_input "$REAPPLY_CHAT_TEMPLATE_TO_INPUT" \
    --top_k_loss "${TOP_K_LOSS:-256}" \
    --jsd_token_clip "${JSD_TOKEN_CLIP:-1e-6}" \
    --wandb_project "$WANDB_PROJECT" \
    "${EXTRA_TRAINING_ARGS[@]}"
