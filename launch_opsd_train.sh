#!/bin/bash

# ----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,5,7}"  # 3 train + 1 vLLM
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-20139}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-false}"
# Sweep overrides (single-GPU idle sweep sets these per experiment).
# Default to the batched Hugging Face generation path. Enable vLLM explicitly
# only when a dedicated generation GPU is available.
USE_VLLM="${USE_VLLM:-false}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_GENERATION_BATCH_SIZE="${TRAIN_GENERATION_BATCH_SIZE:-4}"
TEST_GENERATION_BATCH_SIZE="${TEST_GENERATION_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-ml-1m-128len-opsd-sasrec-na-epoch1-1-seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAINER_MAX_STEPS="${TRAINER_MAX_STEPS:--1}"
EVAL_ON_START="${EVAL_ON_START:-True}"
FIXED_EPOCH_COMBINED_EVAL="${FIXED_EPOCH_COMBINED_EVAL:-True}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SEED="${SEED:-42}"
OPSD_TARGET_WEIGHT="${OPSD_TARGET_WEIGHT:-0.1}"
OPSD_TARGET_LOSS_TYPE="${OPSD_TARGET_LOSS_TYPE:-infonce}"
OPSD_DIRECTAU_WEIGHT="${OPSD_DIRECTAU_WEIGHT:-0.0}"
OPSD_PRIVILEGED_ALIGNMENT_MODE="${OPSD_PRIVILEGED_ALIGNMENT_MODE:-cross_user_item}"
OPSD_PRIVILEGED_ALIGNMENT_WEIGHT="${OPSD_PRIVILEGED_ALIGNMENT_WEIGHT:-1.0}"
OPSD_DIRECTAU_UNIFORM_WEIGHT="${OPSD_DIRECTAU_UNIFORM_WEIGHT:-1.0}"
OPSD_DIRECTAU_UNIFORM_TEMPERATURE="${OPSD_DIRECTAU_UNIFORM_TEMPERATURE:-2.0}"
OPSD_NA_WEIGHT="${OPSD_NA_WEIGHT:-0.1}"
OPSD_TOKEN_WEIGHT="${OPSD_TOKEN_WEIGHT:-1.0}"
OPSD_MAX_TOKEN_KL="${OPSD_MAX_TOKEN_KL:-100.0}"
OPSD_LOSS_TYPE="${OPSD_LOSS_TYPE:-kl}"
OPSD_ENTROPY_GATE_MODE="${OPSD_ENTROPY_GATE_MODE:-none}"
OPSD_ENTROPY_GAP_THRESHOLD="${OPSD_ENTROPY_GAP_THRESHOLD:-0.0}"
OPSD_ENTROPY_GATE_TEMPERATURE="${OPSD_ENTROPY_GATE_TEMPERATURE:-0.1}"
OPSD_ENTROPY_GATE_MIN_TEACHER_TOPK_MASS="${OPSD_ENTROPY_GATE_MIN_TEACHER_TOPK_MASS:-0.9}"
OPSD_ENTROPY_GATE_LOSS_NORMALIZATION="${OPSD_ENTROPY_GATE_LOSS_NORMALIZATION:-active_tokens}"
OPSD_TOPK_MASS_DIAGNOSTICS="${OPSD_TOPK_MASS_DIAGNOSTICS:-false}"
OPSD_TOPK_MASS_MAX_POSITIONS="${OPSD_TOPK_MASS_MAX_POSITIONS:-100000}"
OPSD_GRADIENT_COORDINATION_MODE="${OPSD_GRADIENT_COORDINATION_MODE:-sum}"
OPSD_GRADIENT_COORDINATION_EPSILON="${OPSD_GRADIENT_COORDINATION_EPSILON:-1e-12}"
OPSD_TOKEN_CREDIT_MODE="${OPSD_TOKEN_CREDIT_MODE:-uniform}"
OPSD_TOKEN_CREDIT_TEMPERATURE="${OPSD_TOKEN_CREDIT_TEMPERATURE:-0.2}"
OPSD_TOKEN_CREDIT_MIX="${OPSD_TOKEN_CREDIT_MIX:-1.0}"
OPSD_RATIONALE_GEOMETRY_WEIGHT="${OPSD_RATIONALE_GEOMETRY_WEIGHT:-0.0}"
OPSD_RATIONALE_GEOMETRY_LOSS_TYPE="${OPSD_RATIONALE_GEOMETRY_LOSS_TYPE:-relational_kl}"
OPSD_RATIONALE_REASON_TEMPERATURE="${OPSD_RATIONALE_REASON_TEMPERATURE:-0.2}"
OPSD_RATIONALE_ITEM_TEMPERATURE="${OPSD_RATIONALE_ITEM_TEMPERATURE:-0.2}"
OPSD_PRIVILEGED_CONTEXT="${OPSD_PRIVILEGED_CONTEXT:-target}"
OPSD_HARD_NEGATIVE_COUNT="${OPSD_HARD_NEGATIVE_COUNT:-3}"
OPSD_HARD_NEGATIVE_EXCLUDE_TRAIN_POSITIVES="${OPSD_HARD_NEGATIVE_EXCLUDE_TRAIN_POSITIVES:-True}"
OPSD_RANK_WEIGHT="${OPSD_RANK_WEIGHT:-0.0}"
OPSD_RANK_LOSS_TYPE="${OPSD_RANK_LOSS_TYPE:-distribution_kl}"
OPSD_GEOMETRY_POSITIVE_MARGIN_ONLY="${OPSD_GEOMETRY_POSITIVE_MARGIN_ONLY:-True}"
OPSD_TEACHER_GATE_MODE="${OPSD_TEACHER_GATE_MODE:-hard}"
OPSD_TEACHER_GATE_TEMPERATURE="${OPSD_TEACHER_GATE_TEMPERATURE:-0.1}"
OPSD_TEACHER_GATE_POWER="${OPSD_TEACHER_GATE_POWER:-1.0}"
OPSD_TEACHER_BETTER_ONLY="${OPSD_TEACHER_BETTER_ONLY:-False}"
SASREC_RANK_WEIGHT="${SASREC_RANK_WEIGHT:-0.1}"
SASREC_TEACHER_BETTER_ONLY="${SASREC_TEACHER_BETTER_ONLY:-True}"
SASREC_TEACHER_GATE_MODE="${SASREC_TEACHER_GATE_MODE:-hard}"
SASREC_TEACHER_REWARD_MODE="${SASREC_TEACHER_REWARD_MODE:-legacy_mixed}"
SASREC_TEACHER_GATE_TEMPERATURE="${SASREC_TEACHER_GATE_TEMPERATURE:-0.1}"
SASREC_TEACHER_GATE_POWER="${SASREC_TEACHER_GATE_POWER:-1.0}"
LIGHTGCN_RANK_WEIGHT="${LIGHTGCN_RANK_WEIGHT:-0.0}"
OPSD_SMALL_MOE_WEIGHT="${OPSD_SMALL_MOE_WEIGHT:-0.0}"
OPSD_SMALL_MOE_ROUTER_MODE="${OPSD_SMALL_MOE_ROUTER_MODE:-soft_mixture}"
OPSD_SMALL_MOE_ADVANTAGE_MODE="${OPSD_SMALL_MOE_ADVANTAGE_MODE:-normalized_log_rank}"
OPSD_SMALL_MOE_TOPK="${OPSD_SMALL_MOE_TOPK:-32}"
OPSD_SMALL_MOE_STUDENT_TEMPERATURE="${OPSD_SMALL_MOE_STUDENT_TEMPERATURE:-0.2}"
OPSD_SMALL_MOE_SASREC_TEMPERATURE="${OPSD_SMALL_MOE_SASREC_TEMPERATURE:-0.2}"
OPSD_SMALL_MOE_LIGHTGCN_TEMPERATURE="${OPSD_SMALL_MOE_LIGHTGCN_TEMPERATURE:-0.2}"
OPSD_SMALL_MOE_FMLPREC_TEMPERATURE="${OPSD_SMALL_MOE_FMLPREC_TEMPERATURE:-0.2}"
OPSD_SMALL_MOE_EASE_TEMPERATURE="${OPSD_SMALL_MOE_EASE_TEMPERATURE:-0.2}"
OPSD_SMALL_MOE_GATE_TEMPERATURE="${OPSD_SMALL_MOE_GATE_TEMPERATURE:-0.1}"
OPSD_SMALL_MOE_OUTER_TEMPERATURE="${OPSD_SMALL_MOE_OUTER_TEMPERATURE:-0.1}"
OPSD_SMALL_MOE_QUALITY_POWER="${OPSD_SMALL_MOE_QUALITY_POWER:-1.0}"

# ----------------------------

DATASET_CAT="${DATASET_CAT:-MovieLens_1M}"
DATASET_DIR="${DATASET_DIR:-data/${DATASET_CAT}}"
LIGHTGCN_CHECKPOINT="${LIGHTGCN_CHECKPOINT:-checkpoints/lightgcn_ml1m.pt}"
SASREC_CHECKPOINT="${SASREC_CHECKPOINT:-checkpoints/sasrec_amazon_book_usersample25k.pt}"
SMALL_MOE_SASREC_CHECKPOINT="${SMALL_MOE_SASREC_CHECKPOINT:-$SASREC_CHECKPOINT}"
SMALL_MOE_LIGHTGCN_CHECKPOINT="${SMALL_MOE_LIGHTGCN_CHECKPOINT:-$LIGHTGCN_CHECKPOINT}"
SMALL_MOE_FMLPREC_CHECKPOINT="${SMALL_MOE_FMLPREC_CHECKPOINT:-checkpoints/fmlprec_${DATASET_CAT}.pt}"
SMALL_MOE_EASE_CHECKPOINT="${SMALL_MOE_EASE_CHECKPOINT:-checkpoints/ease_${DATASET_CAT}.pt}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-/home1/workspace/taoruizhi/checkpoints/ml-1m-128len-recpo-epoch1-seed42/final_checkpoint}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/home1/workspace/taoruizhi/checkpoints/ml-1m-128len-recpo-epoch1-seed42/final_checkpoint}"

# Passing the checkpoint below selects a fixed external teacher. Omit it only
# when token/rank distillation and privileged DirectAU are all disabled.

# Run once before OPSD:
# python lightgcn.py --dataset_dir "$DATASET_DIR" \
#     --output_path "$LIGHTGCN_CHECKPOINT" --device cuda:0
# python sasrec.py --dataset_dir "$DATASET_DIR" \
#     --output_path "$SASREC_CHECKPOINT" --device cuda:0

# Full-catalog target CE anchors frozen-teacher token OPSD against collapse.
# USE_VLLM=false appends --use_vllm=False for single-GPU idle-sweep runs
# (no dedicated vLLM card); set USE_VLLM=true only for a dedicated vLLM card.
USE_VLLM_FLAG=()
_shopt_use_vllm="$(echo "$USE_VLLM" | tr '[:upper:]' '[:lower:]')"
if [ "$_shopt_use_vllm" = "false" ] || [ "$_shopt_use_vllm" = "0" ] || [ "$_shopt_use_vllm" = "no" ]; then
                USE_VLLM_FLAG=(--use_vllm=False)
fi

is_zero_weight() {
                awk -v value="$1" 'BEGIN {
                  if (value !~ /^[[:space:]]*[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?[[:space:]]*$/) exit 1
                  exit ((value + 0.0) == 0.0) ? 0 : 1
                }'
}

OPSD_TEACHER_CHECKPOINT_FLAG=(--opsd_teacher_checkpoint="$TEACHER_CHECKPOINT")
_shopt_target_loss_type="${OPSD_TARGET_LOSS_TYPE//-/_}"
_shopt_privileged_directau_enabled=false
if [ "$_shopt_target_loss_type" = "privileged_directau" ] && ! is_zero_weight "$OPSD_TARGET_WEIGHT"; then
                _shopt_privileged_directau_enabled=true
fi
if ! is_zero_weight "$OPSD_DIRECTAU_WEIGHT"; then
                _shopt_privileged_directau_enabled=true
fi
if is_zero_weight "$OPSD_TOKEN_WEIGHT" && is_zero_weight "$OPSD_RANK_WEIGHT"; then
                if [ "$_shopt_privileged_directau_enabled" != "true" ]; then
                                OPSD_TEACHER_CHECKPOINT_FLAG=()
                fi
fi

OPSD_MAX_TOKEN_KL_FLAG=()
if ! is_zero_weight "$OPSD_TOKEN_WEIGHT"; then
                case "${OPSD_MAX_TOKEN_KL,,}" in
                                ""|none|off|false) ;;
                                *) OPSD_MAX_TOKEN_KL_FLAG=(--opsd_max_token_kl="$OPSD_MAX_TOKEN_KL") ;;
                esac
fi

accelerate launch --num_processes="$NUM_PROCESSES" --config_file=accelerates/deepspeed_config.yaml \
                --main_process_port="$MAIN_PROCESS_PORT" opsd_train.py \
                "${USE_VLLM_FLAG[@]}" \
                --model=qwen \
                --dataset_dir="$DATASET_DIR" \
                --dataset_category="$DATASET_CAT" \
                --output_dir="$OUTPUT_DIR" \
                --run_name="$RUN_NAME" \
                --train_batch_size="$TRAIN_BATCH_SIZE" \
                --trainer_mini_batch_size="$MINI_BATCH_SIZE" \
                --num_train_epochs="$NUM_TRAIN_EPOCHS" \
                --eval_batch_size=32 \
                --train_generation_batch_size="$TRAIN_GENERATION_BATCH_SIZE" \
                --test_generation_batch_size="$TEST_GENERATION_BATCH_SIZE" \
                --group_size=1 \
                --max_new_tokens="$MAX_NEW_TOKENS" \
                --opsd_max_length=2048 \
                --opsd_item_bank_refresh_steps=8 \
                --opsd_privileged_context="$OPSD_PRIVILEGED_CONTEXT" \
                --opsd_hard_negative_count="$OPSD_HARD_NEGATIVE_COUNT" \
                --opsd_hard_negative_exclude_train_positives="$OPSD_HARD_NEGATIVE_EXCLUDE_TRAIN_POSITIVES" \
                --opsd_target_weight="$OPSD_TARGET_WEIGHT" \
                --opsd_target_loss_type="$OPSD_TARGET_LOSS_TYPE" \
                --opsd_directau_weight="$OPSD_DIRECTAU_WEIGHT" \
                --opsd_privileged_alignment_mode="$OPSD_PRIVILEGED_ALIGNMENT_MODE" \
                --opsd_privileged_alignment_weight="$OPSD_PRIVILEGED_ALIGNMENT_WEIGHT" \
                --opsd_directau_uniform_weight="$OPSD_DIRECTAU_UNIFORM_WEIGHT" \
                --opsd_directau_uniform_temperature="$OPSD_DIRECTAU_UNIFORM_TEMPERATURE" \
                --opsd_na_weight="$OPSD_NA_WEIGHT" \
                --opsd_na_temperature=0.2 \
                --opsd_token_weight="$OPSD_TOKEN_WEIGHT" \
                --opsd_gradient_coordination_mode="$OPSD_GRADIENT_COORDINATION_MODE" \
                --opsd_gradient_coordination_epsilon="$OPSD_GRADIENT_COORDINATION_EPSILON" \
                --opsd_token_credit_mode="$OPSD_TOKEN_CREDIT_MODE" \
                --opsd_token_credit_temperature="$OPSD_TOKEN_CREDIT_TEMPERATURE" \
                --opsd_token_credit_mix="$OPSD_TOKEN_CREDIT_MIX" \
                --opsd_rationale_geometry_weight="$OPSD_RATIONALE_GEOMETRY_WEIGHT" \
                --opsd_rationale_geometry_loss_type="$OPSD_RATIONALE_GEOMETRY_LOSS_TYPE" \
                --opsd_rationale_reason_temperature="$OPSD_RATIONALE_REASON_TEMPERATURE" \
                --opsd_rationale_item_temperature="$OPSD_RATIONALE_ITEM_TEMPERATURE" \
                --opsd_rank_weight="$OPSD_RANK_WEIGHT" \
                --opsd_rank_loss_type="$OPSD_RANK_LOSS_TYPE" \
                --opsd_geometry_positive_margin_only="$OPSD_GEOMETRY_POSITIVE_MARGIN_ONLY" \
                --opsd_rank_temperature=1.0 \
                --opsd_rank_topk=64 \
                --opsd_teacher_better_only="$OPSD_TEACHER_BETTER_ONLY" \
                --opsd_teacher_gate_mode="$OPSD_TEACHER_GATE_MODE" \
                --opsd_teacher_gate_temperature="$OPSD_TEACHER_GATE_TEMPERATURE" \
                --opsd_teacher_gate_power="$OPSD_TEACHER_GATE_POWER" \
                "${OPSD_TEACHER_CHECKPOINT_FLAG[@]}" \
                --opsd_teacher_sync_steps=128 \
                --opsd_temperature=1.0 \
                "${OPSD_MAX_TOKEN_KL_FLAG[@]}" \
                --opsd_loss_type="$OPSD_LOSS_TYPE" \
                --opsd_topk=1024 \
                --opsd_entropy_gate_mode="$OPSD_ENTROPY_GATE_MODE" \
                --opsd_entropy_gap_threshold="$OPSD_ENTROPY_GAP_THRESHOLD" \
                --opsd_entropy_gate_temperature="$OPSD_ENTROPY_GATE_TEMPERATURE" \
                --opsd_entropy_gate_min_teacher_topk_mass="$OPSD_ENTROPY_GATE_MIN_TEACHER_TOPK_MASS" \
                --opsd_entropy_gate_loss_normalization="$OPSD_ENTROPY_GATE_LOSS_NORMALIZATION" \
                --opsd_topk_mass_diagnostics="$OPSD_TOPK_MASS_DIAGNOSTICS" \
                --opsd_topk_mass_max_positions="$OPSD_TOPK_MASS_MAX_POSITIONS" \
                --opsd_token_chunk_size=64 \
                --lightgcn_checkpoint="$LIGHTGCN_CHECKPOINT" \
                --lightgcn_rank_weight="$LIGHTGCN_RANK_WEIGHT" \
                --lightgcn_topk=64 \
                --lightgcn_teacher_temperature=0.2 \
                --lightgcn_student_temperature=0.2 \
                --lightgcn_teacher_better_only=True \
                --sasrec_checkpoint="$SASREC_CHECKPOINT" \
                --sasrec_rank_weight="$SASREC_RANK_WEIGHT" \
                --sasrec_topk=64 \
                --sasrec_teacher_temperature=0.2 \
                --sasrec_student_temperature=0.2 \
                --sasrec_teacher_better_only="$SASREC_TEACHER_BETTER_ONLY" \
                --sasrec_teacher_gate_mode="$SASREC_TEACHER_GATE_MODE" \
                --sasrec_teacher_reward_mode="$SASREC_TEACHER_REWARD_MODE" \
                --sasrec_teacher_gate_temperature="$SASREC_TEACHER_GATE_TEMPERATURE" \
                --sasrec_teacher_gate_power="$SASREC_TEACHER_GATE_POWER" \
                --opsd_small_moe_weight="$OPSD_SMALL_MOE_WEIGHT" \
                --opsd_small_moe_router_mode="$OPSD_SMALL_MOE_ROUTER_MODE" \
                --opsd_small_moe_advantage_mode="$OPSD_SMALL_MOE_ADVANTAGE_MODE" \
                --small_moe_sasrec_checkpoint="$SMALL_MOE_SASREC_CHECKPOINT" \
                --small_moe_lightgcn_checkpoint="$SMALL_MOE_LIGHTGCN_CHECKPOINT" \
                --small_moe_fmlprec_checkpoint="$SMALL_MOE_FMLPREC_CHECKPOINT" \
                --small_moe_ease_checkpoint="$SMALL_MOE_EASE_CHECKPOINT" \
                --opsd_small_moe_topk="$OPSD_SMALL_MOE_TOPK" \
                --opsd_small_moe_student_temperature="$OPSD_SMALL_MOE_STUDENT_TEMPERATURE" \
                --opsd_small_moe_sasrec_temperature="$OPSD_SMALL_MOE_SASREC_TEMPERATURE" \
                --opsd_small_moe_lightgcn_temperature="$OPSD_SMALL_MOE_LIGHTGCN_TEMPERATURE" \
                --opsd_small_moe_fmlprec_temperature="$OPSD_SMALL_MOE_FMLPREC_TEMPERATURE" \
                --opsd_small_moe_ease_temperature="$OPSD_SMALL_MOE_EASE_TEMPERATURE" \
                --opsd_small_moe_gate_temperature="$OPSD_SMALL_MOE_GATE_TEMPERATURE" \
                --opsd_small_moe_outer_temperature="$OPSD_SMALL_MOE_OUTER_TEMPERATURE" \
                --opsd_small_moe_quality_power="$OPSD_SMALL_MOE_QUALITY_POWER" \
                --trainer_max_steps="$TRAINER_MAX_STEPS" \
                --eval_freq=32 \
                --early_stopping_patience=32 \
                --eval_on_start="$EVAL_ON_START" \
                --fixed_epoch_combined_eval="$FIXED_EPOCH_COMBINED_EVAL" \
                --resume_from_checkpoint="$RESUME_FROM_CHECKPOINT" \
                --seed="$SEED" \
                --trainer_vllm_gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
                --init_checkpoint="$STUDENT_CHECKPOINT"

# sasrec训练命令
# python sasrec.py --dataset_dir data/Amazon_Books_UserSample25K --output_path checkpoints/sasrec_amazon_book_usersample25k.pt --hidden_dim 64 --num_blocks 2 --max_seq_len 50 --num_heads 2 --epochs 200 --batch_size 256 --lr 0.001 --valid_k 10 --eval_every 5 --device cuda:0 --seed 42
# python lightgcn.py --dataset_dir data/Amazon_Books_UserSample25K --output_path checkpoints/lightgcn_amazon_book_usersample25k.pt --seed 42 --n_layers 3 --epochs 500 --emb_dim 64 --valid_k 10 --device cuda:0
