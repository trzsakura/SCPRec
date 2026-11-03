"""Command-line entry point for OPSD without changing the original train.py."""

import fire

import train as original_train
from small_teacher_moe import (
    normalize_small_moe_advantage_mode,
    normalize_small_moe_router_mode,
)
from trainers.opsd_trainer import (
    OPSDOutcomeGRPOTrainer,
    OPSDTrainingArguments,
    _normalize_opsd_gradient_coordination_mode,
    _normalize_opsd_entropy_gate_loss_normalization,
    _normalize_opsd_privileged_alignment_mode,
    _normalize_opsd_rationale_geometry_loss_type,
    _normalize_opsd_rank_loss_type,
    _normalize_opsd_target_loss_type,
    _normalize_opsd_teacher_gate_mode,
    _normalize_opsd_token_credit_mode,
    _resolve_opsd_teacher_update_mode,
)


# Reuse the project's original model/data/optimizer setup.  The wrapper only
# swaps the Trainer and its argument dataclass; train.py itself is untouched.
original_train.OutcomeGRPOTrainer = OPSDOutcomeGRPOTrainer
original_train.OutcomeGRPOTrainingArguments = OPSDTrainingArguments


def _coerce_bool_argument(value, name):
    """Parse Fire/shell boolean values without relying on string truthiness."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value; got {value!r}."
    )


def _normalize_resume_from_checkpoint_argument(value):
    """Normalize shell booleans while preserving a real checkpoint path."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip()
    lowered = normalized.lower()
    if lowered in {"", "0", "false", "none", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return normalized


def train(
        opsd_target_weight=0.1,
        opsd_target_loss_type="full_softmax_ce",
        opsd_directau_weight=0.0,
        opsd_privileged_alignment_mode="cross_user_item",
        opsd_privileged_alignment_weight=1.0,
        opsd_directau_uniform_weight=1.0,
        opsd_directau_uniform_temperature=2.0,
        opsd_na_weight=0.1,
        opsd_na_temperature=0.2,
        opsd_token_weight=1.0,
        opsd_gradient_coordination_mode="sum",
        opsd_gradient_coordination_epsilon=1e-12,
        opsd_token_credit_mode="uniform",
        opsd_token_credit_temperature=0.2,
        opsd_token_credit_mix=1.0,
        opsd_rationale_geometry_weight=0.0,
        opsd_rationale_geometry_loss_type="relational_kl",
        opsd_rationale_reason_temperature=0.2,
        opsd_rationale_item_temperature=0.2,
        opsd_rank_weight=0.0,
        opsd_rank_loss_type="distribution_kl",
        opsd_geometry_positive_margin_only=True,
        opsd_rank_temperature=1.0,
        opsd_rank_topk=64,
        opsd_teacher_better_only=False,
        opsd_teacher_gate_mode="hard",
        opsd_teacher_gate_temperature=0.1,
        opsd_teacher_gate_power=1.0,
        opsd_frozen_teacher=False,
        opsd_teacher_checkpoint=None,
        opsd_teacher_update_mode="auto",
        opsd_teacher_sync_steps=128,
        opsd_embed_weight=None,
        opsd_temperature=1.0,
        opsd_max_token_kl=None,
        opsd_loss_type="kl",
        opsd_topk=64,
        opsd_entropy_gate_mode="none",
        opsd_entropy_gap_threshold=0.0,
        opsd_entropy_gate_temperature=0.1,
        opsd_entropy_gate_min_teacher_topk_mass=0.0,
        opsd_entropy_gate_loss_normalization="active_tokens",
        opsd_topk_mass_diagnostics=False,
        opsd_topk_mass_max_positions=100000,
        opsd_token_chunk_size=64,
        opsd_max_length=None,
        opsd_privileged_context="target",
        opsd_hard_negative_count=3,
        opsd_hard_negative_exclude_train_positives=True,
        opsd_item_bank_refresh_steps=-1,
        lightgcn_checkpoint=None,
        lightgcn_rank_weight=0.0,
        lightgcn_topk=32,
        lightgcn_teacher_temperature=0.2,
        lightgcn_student_temperature=0.2,
        lightgcn_teacher_better_only=True,
        sasrec_checkpoint=None,
        sasrec_rank_weight=0.0,
        sasrec_topk=32,
        sasrec_teacher_temperature=0.2,
        sasrec_student_temperature=0.2,
        sasrec_teacher_better_only=True,
        sasrec_teacher_gate_mode="hard",
        sasrec_teacher_reward_mode="normalized_log_rank",
        sasrec_teacher_gate_temperature=0.1,
        sasrec_teacher_gate_power=1.0,
        opsd_small_moe_weight=0.0,
        opsd_small_moe_router_mode="soft_mixture",
        opsd_small_moe_advantage_mode="target_rank",
        small_moe_sasrec_checkpoint=None,
        small_moe_lightgcn_checkpoint=None,
        small_moe_fmlprec_checkpoint=None,
        small_moe_ease_checkpoint=None,
        opsd_small_moe_topk=32,
        opsd_small_moe_student_temperature=0.2,
        opsd_small_moe_sasrec_temperature=0.2,
        opsd_small_moe_lightgcn_temperature=0.2,
        opsd_small_moe_fmlprec_temperature=0.2,
        opsd_small_moe_ease_temperature=0.2,
        opsd_small_moe_gate_temperature=0.1,
        opsd_small_moe_outer_temperature=0.1,
        opsd_small_moe_quality_power=1.0,
        fixed_epoch_combined_eval=False,
        eval_on_start=True,
        trainer_max_steps=-1,
        resume_from_checkpoint=False,
        init_checkpoint=None,
        **kwargs,
):
    """Run the original Recommendation entry point with strict OPSD enabled.

    All original train.py arguments remain available.  OPSD arguments are
    forwarded with the project's existing ``trainer_`` convention.
    """

    opsd_topk_mass_diagnostics = _coerce_bool_argument(
        opsd_topk_mass_diagnostics,
        "opsd_topk_mass_diagnostics",
    )
    fixed_epoch_combined_eval = _coerce_bool_argument(
        fixed_epoch_combined_eval,
        "fixed_epoch_combined_eval",
    )
    eval_on_start = _coerce_bool_argument(
        eval_on_start,
        "eval_on_start",
    )
    resume_from_checkpoint = _normalize_resume_from_checkpoint_argument(
        resume_from_checkpoint
    )

    if opsd_embed_weight is not None:
        # Backward compatibility for launch commands created before the
        # recommendation-space loss replaced hidden-state cosine alignment.
        opsd_rank_weight = opsd_embed_weight

    if opsd_teacher_checkpoint is not None:
        opsd_teacher_checkpoint = str(opsd_teacher_checkpoint).strip()
        if not opsd_teacher_checkpoint:
            raise ValueError("opsd_teacher_checkpoint must not be empty.")
        if kwargs.get("use_lora", True) is False:
            raise ValueError(
                "opsd_teacher_checkpoint expects a PEFT/LoRA checkpoint, "
                "so use_lora must be True."
            )
    opsd_target_loss_type = _normalize_opsd_target_loss_type(
        opsd_target_loss_type
    )
    opsd_privileged_alignment_mode = (
        _normalize_opsd_privileged_alignment_mode(
            opsd_privileged_alignment_mode
        )
    )
    opsd_teacher_update_mode = _resolve_opsd_teacher_update_mode(
        mode=opsd_teacher_update_mode,
        teacher_checkpoint=opsd_teacher_checkpoint,
        token_weight=float(opsd_token_weight),
        rank_weight=float(opsd_rank_weight),
        target_weight=float(opsd_target_weight),
        target_loss_type=opsd_target_loss_type,
        directau_weight=float(opsd_directau_weight),
    )
    # Frozen means independent parameter storage between synchronization
    # points. ``live`` is retained only as an explicit compatibility mode.
    opsd_frozen_teacher = opsd_teacher_update_mode in {"fixed", "lagged"}

    kwargs["trainer_opsd_target_weight"] = opsd_target_weight
    kwargs["trainer_opsd_target_loss_type"] = opsd_target_loss_type
    kwargs["trainer_opsd_directau_weight"] = opsd_directau_weight
    kwargs["trainer_opsd_privileged_alignment_mode"] = (
        opsd_privileged_alignment_mode
    )
    kwargs["trainer_opsd_privileged_alignment_weight"] = (
        opsd_privileged_alignment_weight
    )
    kwargs["trainer_opsd_directau_uniform_weight"] = (
        opsd_directau_uniform_weight
    )
    kwargs["trainer_opsd_directau_uniform_temperature"] = (
        opsd_directau_uniform_temperature
    )
    kwargs["trainer_opsd_na_weight"] = opsd_na_weight
    kwargs["trainer_opsd_na_temperature"] = opsd_na_temperature
    kwargs["trainer_opsd_token_weight"] = opsd_token_weight
    kwargs["trainer_opsd_gradient_coordination_mode"] = (
        _normalize_opsd_gradient_coordination_mode(
            opsd_gradient_coordination_mode
        )
    )
    kwargs["trainer_opsd_gradient_coordination_epsilon"] = (
        opsd_gradient_coordination_epsilon
    )
    kwargs["trainer_opsd_token_credit_mode"] = (
        _normalize_opsd_token_credit_mode(opsd_token_credit_mode)
    )
    kwargs["trainer_opsd_token_credit_temperature"] = (
        opsd_token_credit_temperature
    )
    kwargs["trainer_opsd_token_credit_mix"] = opsd_token_credit_mix
    kwargs["trainer_opsd_rationale_geometry_weight"] = (
        opsd_rationale_geometry_weight
    )
    kwargs["trainer_opsd_rationale_geometry_loss_type"] = (
        _normalize_opsd_rationale_geometry_loss_type(
            opsd_rationale_geometry_loss_type
        )
    )
    kwargs["trainer_opsd_rationale_reason_temperature"] = (
        opsd_rationale_reason_temperature
    )
    kwargs["trainer_opsd_rationale_item_temperature"] = (
        opsd_rationale_item_temperature
    )
    kwargs["trainer_opsd_rank_weight"] = opsd_rank_weight
    kwargs["trainer_opsd_rank_loss_type"] = (
        _normalize_opsd_rank_loss_type(opsd_rank_loss_type)
    )
    kwargs["trainer_opsd_geometry_positive_margin_only"] = (
        opsd_geometry_positive_margin_only
    )
    kwargs["trainer_opsd_rank_temperature"] = opsd_rank_temperature
    kwargs["trainer_opsd_rank_topk"] = opsd_rank_topk
    kwargs["trainer_opsd_teacher_better_only"] = opsd_teacher_better_only
    kwargs["trainer_opsd_teacher_gate_mode"] = (
        _normalize_opsd_teacher_gate_mode(opsd_teacher_gate_mode)
    )
    kwargs["trainer_opsd_teacher_gate_temperature"] = (
        opsd_teacher_gate_temperature
    )
    kwargs["trainer_opsd_teacher_gate_power"] = opsd_teacher_gate_power
    kwargs["trainer_opsd_frozen_teacher"] = opsd_frozen_teacher
    kwargs["trainer_opsd_teacher_checkpoint"] = opsd_teacher_checkpoint
    kwargs["trainer_opsd_teacher_update_mode"] = opsd_teacher_update_mode
    kwargs["trainer_opsd_teacher_sync_steps"] = opsd_teacher_sync_steps
    kwargs["trainer_opsd_temperature"] = opsd_temperature
    kwargs["trainer_opsd_max_token_kl"] = opsd_max_token_kl
    kwargs["trainer_opsd_loss_type"] = opsd_loss_type
    kwargs["trainer_opsd_topk"] = opsd_topk
    kwargs["trainer_opsd_entropy_gate_mode"] = opsd_entropy_gate_mode
    kwargs["trainer_opsd_entropy_gap_threshold"] = (
        opsd_entropy_gap_threshold
    )
    kwargs["trainer_opsd_entropy_gate_temperature"] = (
        opsd_entropy_gate_temperature
    )
    kwargs["trainer_opsd_entropy_gate_min_teacher_topk_mass"] = (
        opsd_entropy_gate_min_teacher_topk_mass
    )
    kwargs["trainer_opsd_entropy_gate_loss_normalization"] = (
        _normalize_opsd_entropy_gate_loss_normalization(
            opsd_entropy_gate_loss_normalization
        )
    )
    kwargs["trainer_opsd_topk_mass_diagnostics"] = (
        opsd_topk_mass_diagnostics
    )
    kwargs["trainer_opsd_topk_mass_max_positions"] = (
        opsd_topk_mass_max_positions
    )
    kwargs["trainer_opsd_token_chunk_size"] = opsd_token_chunk_size
    kwargs["trainer_opsd_max_length"] = opsd_max_length
    kwargs["trainer_opsd_privileged_context"] = opsd_privileged_context
    kwargs["trainer_opsd_hard_negative_count"] = (
        opsd_hard_negative_count
    )
    kwargs["trainer_opsd_hard_negative_exclude_train_positives"] = (
        opsd_hard_negative_exclude_train_positives
    )
    kwargs["trainer_opsd_item_bank_refresh_steps"] = (
        opsd_item_bank_refresh_steps
    )
    kwargs["trainer_lightgcn_checkpoint"] = lightgcn_checkpoint
    kwargs["trainer_lightgcn_rank_weight"] = lightgcn_rank_weight
    kwargs["trainer_lightgcn_topk"] = lightgcn_topk
    kwargs["trainer_lightgcn_teacher_temperature"] = (
        lightgcn_teacher_temperature
    )
    kwargs["trainer_lightgcn_student_temperature"] = (
        lightgcn_student_temperature
    )
    kwargs["trainer_lightgcn_teacher_better_only"] = (
        lightgcn_teacher_better_only
    )
    kwargs["trainer_sasrec_checkpoint"] = sasrec_checkpoint
    kwargs["trainer_sasrec_rank_weight"] = sasrec_rank_weight
    kwargs["trainer_sasrec_topk"] = sasrec_topk
    kwargs["trainer_sasrec_teacher_temperature"] = (
        sasrec_teacher_temperature
    )
    kwargs["trainer_sasrec_student_temperature"] = (
        sasrec_student_temperature
    )
    kwargs["trainer_sasrec_teacher_better_only"] = (
        sasrec_teacher_better_only
    )
    kwargs["trainer_sasrec_teacher_gate_mode"] = sasrec_teacher_gate_mode
    kwargs["trainer_sasrec_teacher_reward_mode"] = (
        sasrec_teacher_reward_mode
    )
    kwargs["trainer_sasrec_teacher_gate_temperature"] = (
        sasrec_teacher_gate_temperature
    )
    kwargs["trainer_sasrec_teacher_gate_power"] = sasrec_teacher_gate_power
    kwargs["trainer_opsd_small_moe_weight"] = opsd_small_moe_weight
    kwargs["trainer_opsd_small_moe_router_mode"] = (
        normalize_small_moe_router_mode(opsd_small_moe_router_mode)
    )
    kwargs["trainer_opsd_small_moe_advantage_mode"] = (
        normalize_small_moe_advantage_mode(opsd_small_moe_advantage_mode)
    )
    kwargs["trainer_small_moe_sasrec_checkpoint"] = (
        small_moe_sasrec_checkpoint
    )
    kwargs["trainer_small_moe_lightgcn_checkpoint"] = (
        small_moe_lightgcn_checkpoint
    )
    kwargs["trainer_small_moe_fmlprec_checkpoint"] = (
        small_moe_fmlprec_checkpoint
    )
    kwargs["trainer_small_moe_ease_checkpoint"] = small_moe_ease_checkpoint
    kwargs["trainer_opsd_small_moe_topk"] = opsd_small_moe_topk
    kwargs["trainer_opsd_small_moe_student_temperature"] = (
        opsd_small_moe_student_temperature
    )
    kwargs["trainer_opsd_small_moe_sasrec_temperature"] = (
        opsd_small_moe_sasrec_temperature
    )
    kwargs["trainer_opsd_small_moe_lightgcn_temperature"] = (
        opsd_small_moe_lightgcn_temperature
    )
    kwargs["trainer_opsd_small_moe_fmlprec_temperature"] = (
        opsd_small_moe_fmlprec_temperature
    )
    kwargs["trainer_opsd_small_moe_ease_temperature"] = (
        opsd_small_moe_ease_temperature
    )
    kwargs["trainer_opsd_small_moe_gate_temperature"] = (
        opsd_small_moe_gate_temperature
    )
    kwargs["trainer_opsd_small_moe_outer_temperature"] = (
        opsd_small_moe_outer_temperature
    )
    kwargs["trainer_opsd_small_moe_quality_power"] = (
        opsd_small_moe_quality_power
    )
    # These must be explicit Fire parameters. Otherwise values such as
    # "false" can remain non-empty strings in **kwargs and become truthy in
    # train.py, silently re-enabling the fixed-epoch protocol.
    kwargs["fixed_epoch_combined_eval"] = fixed_epoch_combined_eval
    kwargs["eval_on_start"] = eval_on_start
    kwargs["trainer_max_steps"] = trainer_max_steps
    kwargs["resume_from_checkpoint"] = resume_from_checkpoint
    # ``init_checkpoint`` is consumed here to replace ``get_peft_model``, but
    # train.py also needs the original path for evaluation-only validation and
    # for its checkpoint evaluation protocol.  Do not drop it at this wrapper.
    kwargs["init_checkpoint"] = init_checkpoint

    # The original entry point applies a fresh LoRA adapter after loading the
    # base model.  When an existing Recommendation final_checkpoint is supplied, replace
    # only that function call so the saved adapter is loaded as trainable.
    original_get_peft_model = original_train.get_peft_model
    if init_checkpoint is not None:
        if kwargs.get("use_lora", True) is False:
            raise ValueError(
                "init_checkpoint expects a PEFT/LoRA checkpoint, so use_lora must be True."
            )
        from peft import PeftModel

        def _load_existing_adapter(base_model, _peft_config):
            return PeftModel.from_pretrained(
                base_model,
                init_checkpoint,
                is_trainable=True,
            )

        original_train.get_peft_model = _load_existing_adapter

    try:
        return original_train.train(**kwargs)
    finally:
        original_train.get_peft_model = original_get_peft_model


if __name__ == "__main__":
    fire.Fire(train)
