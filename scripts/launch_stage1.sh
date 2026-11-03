#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"   #0,1,2,3
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

NUM_PROCESSES="${NUM_PROCESSES:-3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-20138}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints}"
RUN_NAME="${RUN_NAME:-ml-1m-128len-outcome_grpo-epoch1-seed42}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GENERATION_BATCH_SIZE="${TRAIN_GENERATION_BATCH_SIZE:-16}"
TEST_GENERATION_BATCH_SIZE="${TEST_GENERATION_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
SEED="${SEED:-42}"
USE_VLLM="${USE_VLLM:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-false}"

# ----------------------------
#DATASET_CAT='Amazon_Books' # 'MovieLens_1M'
#DATASET_DIR='data/Amazon_Books_UserSample25K' # "data/${DATASET_CAT}"
DATASET_CAT="${DATASET_CAT:-MovieLens_1M}"
DATASET_DIR="${DATASET_DIR:-data/${DATASET_CAT}}"
# DATASET_CAT='CDs_and_Vinyl'
# DATASET_DIR='data/'$DATASET_CAT'_0_2022-10-2023-10'

USE_VLLM_FLAG=()
_shopt_use_vllm="$(echo "$USE_VLLM" | tr '[:upper:]' '[:lower:]')"
if [ "$_shopt_use_vllm" = "false" ] || [ "$_shopt_use_vllm" = "0" ] || [ "$_shopt_use_vllm" = "no" ]; then
                USE_VLLM_FLAG=(--use_vllm=False)
fi

accelerate launch --num_processes="$NUM_PROCESSES" --config_file=accelerates/deepspeed_config.yaml \
                --main_process_port="$MAIN_PROCESS_PORT" train.py \
                "${USE_VLLM_FLAG[@]}" \
                --model=qwen \
                --model_name_or_path="$MODEL_NAME_OR_PATH" \
                --dataset_dir="$DATASET_DIR" \
                --dataset_category="$DATASET_CAT" \
                --output_dir="$OUTPUT_DIR" \
                --train_batch_size="$TRAIN_BATCH_SIZE" \
                --trainer_mini_batch_size="$MINI_BATCH_SIZE" \
                --eval_batch_size=32 \
                --train_generation_batch_size="$TRAIN_GENERATION_BATCH_SIZE" \
                --test_generation_batch_size="$TEST_GENERATION_BATCH_SIZE" \
                --max_new_tokens="$MAX_NEW_TOKENS" \
                --warmup_steps=32 \
                --seed="$SEED" \
                --num_train_epochs="$NUM_TRAIN_EPOCHS" \
                --fixed_epoch_combined_eval=True \
                --run_name="$RUN_NAME" \
                --group_size=4 \
                --trainer_relabel_topk=2 \
                --gen_temperature=2.0 \
                --gen_top_k=50 \
                --eval_freq=8 \
                --early_stopping_patience=320 \
		--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT"
