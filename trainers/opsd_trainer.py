"""On-policy self-distillation for the existing Recommendation training pipeline.

This module deliberately reuses the original Recommendation data collator and
``OutcomeGRPOTrainer``.  The teacher batch is built after the student rollout,
because the strict OPSD target is the student's own sampled completion.
"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object

from lightgcn import (
    LightGCNEmbeddingBank,
    lightgcn_rank_distillation_loss,
)
from sasrec import (
    SASRecTeacher,
    _normalize_sasrec_teacher_gate_mode,
    _normalize_sasrec_teacher_reward_mode,
    sasrec_rank_distillation_loss,
)
from small_teacher_moe import (
    SmallTeacherMoE,
    normalize_small_moe_advantage_mode,
    normalize_small_moe_router_mode,
    small_teacher_moe_distillation_loss,
)
from prompters.recommendation_prompter import get_item_info_str, get_items_str
from trainers.OutcomeGRPOTrainer import OutcomeGRPOTrainer, OutcomeGRPOTrainingArguments


_OPSD_TEACHER_ADAPTER_NAME = "opsd_privileged_teacher"
_OPSD_TEACHER_STATE_FILENAME = "opsd_teacher_state.pt"
_OPSD_TEACHER_UPDATE_MODES = {"auto", "fixed", "lagged", "live"}
_OPSD_TARGET_LOSS_TYPES = {
    "infonce",
    "full_softmax_ce",
    "privileged_directau",
}
_OPSD_PRIVILEGED_ALIGNMENT_MODES = {
    "cross_user_item",
    "user_to_user",
}
_OPSD_TOKEN_CREDIT_MODES = {
    "uniform",
    "target_item_cosine",
    "teacher_last_layer_attention",
}
_OPSD_ENTROPY_GATE_MODES = {"none", "hard", "soft"}
_OPSD_ENTROPY_GATE_LOSS_NORMALIZATIONS = {
    "active_tokens",
    "candidate_tokens",
}
_OPSD_RATIONALE_GEOMETRY_LOSS_TYPES = {
    "relational_kl",
    "correlation",
}
_OPSD_RANK_LOSS_TYPES = {"distribution_kl", "geometry_margin"}
_OPSD_TEACHER_GATE_MODES = {"hard", "advantage_weighted"}
_OPSD_GRADIENT_COORDINATION_MODES = {
    "sum",
    "diagnostic_only",
    "rec_preserving_projection",
}


def _normalize_opsd_target_loss_type(loss_type: str) -> str:
    """Normalize the selectable ground-truth recommendation objective."""

    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized not in _OPSD_TARGET_LOSS_TYPES:
        choices = ", ".join(sorted(_OPSD_TARGET_LOSS_TYPES))
        raise ValueError(
            f"opsd_target_loss_type must be one of: {choices}."
        )
    return normalized


def _normalize_opsd_privileged_alignment_mode(mode: str) -> str:
    """Normalize the privileged DirectAU alignment structure."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_PRIVILEGED_ALIGNMENT_MODES:
        choices = ", ".join(sorted(_OPSD_PRIVILEGED_ALIGNMENT_MODES))
        raise ValueError(
            "opsd_privileged_alignment_mode must be one of: "
            f"{choices}."
        )
    return normalized


def _normalize_opsd_token_credit_mode(mode: str) -> str:
    """Normalize the token-credit policy used by token-level OPSD."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_TOKEN_CREDIT_MODES:
        choices = ", ".join(sorted(_OPSD_TOKEN_CREDIT_MODES))
        raise ValueError(
            f"opsd_token_credit_mode must be one of: {choices}."
        )
    return normalized


def _normalize_opsd_entropy_gate_mode(mode: str) -> str:
    """Normalize the token-level entropy-difference selection policy."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_ENTROPY_GATE_MODES:
        choices = ", ".join(sorted(_OPSD_ENTROPY_GATE_MODES))
        raise ValueError(
            f"opsd_entropy_gate_mode must be one of: {choices}."
        )
    return normalized


def _normalize_opsd_entropy_gate_loss_normalization(mode: str) -> str:
    """Normalize the denominator used after entropy token selection."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_ENTROPY_GATE_LOSS_NORMALIZATIONS:
        choices = ", ".join(sorted(
            _OPSD_ENTROPY_GATE_LOSS_NORMALIZATIONS
        ))
        raise ValueError(
            "opsd_entropy_gate_loss_normalization must be one of: "
            f"{choices}."
        )
    return normalized


def _reduce_entropy_gated_token_loss(
        per_token_loss: torch.Tensor,
        keep_mask: torch.Tensor,
        normalization: str,
):
    """Reduce retained token losses with an explicit, auditable denominator."""

    if per_token_loss.ndim != 1 or keep_mask.ndim != 1:
        raise ValueError("Entropy-gated losses and mask must be vectors.")
    if per_token_loss.shape != keep_mask.shape:
        raise ValueError("Entropy-gated losses and mask must have equal shape.")
    return _reduce_entropy_weighted_token_loss(
        per_token_loss=per_token_loss,
        token_weights=keep_mask.detach().to(dtype=per_token_loss.dtype),
        normalization=normalization,
    )


def _reduce_entropy_weighted_token_loss(
        per_token_loss: torch.Tensor,
        token_weights: torch.Tensor,
        normalization: str,
):
    """Reduce token losses using detached hard or soft entropy weights."""

    normalization = _normalize_opsd_entropy_gate_loss_normalization(
        normalization
    )
    if per_token_loss.ndim != 1 or token_weights.ndim != 1:
        raise ValueError("Entropy-gated losses and weights must be vectors.")
    if per_token_loss.shape != token_weights.shape:
        raise ValueError(
            "Entropy-gated losses and weights must have equal shape."
        )
    weights = token_weights.detach().to(
        device=per_token_loss.device,
        dtype=per_token_loss.dtype,
    )
    if not torch.isfinite(weights).all():
        raise ValueError("Entropy-gate weights must be finite.")
    if ((weights < 0) | (weights > 1)).any():
        raise ValueError("Entropy-gate weights must be in [0, 1].")
    if per_token_loss.numel() == 0:
        return per_token_loss[:0].sum()

    weighted_sum = (weights * per_token_loss).sum()
    if normalization == "candidate_tokens":
        return weighted_sum / per_token_loss.numel()
    weight_sum = weights.sum()
    safe_weight_sum = torch.where(
        weight_sum > 0,
        weight_sum,
        torch.ones_like(weight_sum),
    )
    return weighted_sum / safe_weight_sum


def _normalize_opsd_rationale_geometry_loss_type(loss_type: str) -> str:
    """Normalize the rationale-to-item relation matching objective."""

    normalized = str(loss_type).strip().lower().replace("-", "_")
    aliases = {
        "kl": "relational_kl",
        "corr": "correlation",
        "pearson": "correlation",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _OPSD_RATIONALE_GEOMETRY_LOSS_TYPES:
        choices = ", ".join(sorted(_OPSD_RATIONALE_GEOMETRY_LOSS_TYPES))
        raise ValueError(
            "opsd_rationale_geometry_loss_type must be one of: "
            f"{choices}."
        )
    return normalized


def _normalize_opsd_teacher_gate_mode(mode: str) -> str:
    """Normalize the per-example privileged-teacher trust policy."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_TEACHER_GATE_MODES:
        choices = ", ".join(sorted(_OPSD_TEACHER_GATE_MODES))
        raise ValueError(
            f"opsd_teacher_gate_mode must be one of: {choices}."
        )
    return normalized


def _normalize_opsd_gradient_coordination_mode(mode: str) -> str:
    """Normalize how recommendation and token-KD gradients interact."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _OPSD_GRADIENT_COORDINATION_MODES:
        choices = ", ".join(sorted(_OPSD_GRADIENT_COORDINATION_MODES))
        raise ValueError(
            "opsd_gradient_coordination_mode must be one of: "
            f"{choices}."
        )
    return normalized


def _recommendation_preserving_gradient_surrogate(
        recommendation_loss: torch.Tensor,
        text_loss: torch.Tensor,
        parameters,
        mode: str,
        epsilon: float,
        gradient_reducer=None,
):
    """Build a scalar loss whose backward protects recommendation progress.

    The optional projection removes only the component of the token-KD
    gradient that opposes the recommendation-space gradient.  The returned
    scalar has exactly the same forward value as the ordinary additive loss,
    so reporting and scheduling remain comparable with the legacy path.
    """

    mode = _normalize_opsd_gradient_coordination_mode(mode)
    additive_loss = recommendation_loss + text_loss
    if mode == "sum":
        # This early return is intentional: the compatibility path must not
        # enumerate parameters or construct any additional autograd graph.
        return additive_loss, {}

    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(
            "opsd_gradient_coordination_epsilon must be finite and positive."
        )
    if recommendation_loss.ndim != 0 or text_loss.ndim != 0:
        raise ValueError(
            "Gradient coordination requires scalar recommendation and text "
            "losses."
        )
    if not recommendation_loss.requires_grad or not text_loss.requires_grad:
        raise ValueError(
            "Gradient coordination losses must remain attached to autograd."
        )

    trainable_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.requires_grad
    )
    if not trainable_parameters:
        raise ValueError(
            "Gradient coordination found no trainable student parameters."
        )

    recommendation_gradients = torch.autograd.grad(
        recommendation_loss,
        trainable_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    text_gradients = torch.autograd.grad(
        text_loss,
        trainable_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def _flatten_gradients(gradients):
        chunks = []
        for parameter, gradient in zip(trainable_parameters, gradients):
            if gradient is None:
                chunk = torch.zeros(
                    parameter.numel(),
                    device=parameter.device,
                    dtype=torch.float32,
                )
            else:
                if gradient.is_sparse:
                    gradient = gradient.to_dense()
                chunk = gradient.detach().to(dtype=torch.float32).reshape(-1)
            chunks.append(chunk)
        return torch.cat(chunks)

    flat_recommendation_gradient = _flatten_gradients(
        recommendation_gradients
    )
    flat_text_gradient = _flatten_gradients(text_gradients)
    if gradient_reducer is not None:
        flat_recommendation_gradient = gradient_reducer(
            flat_recommendation_gradient
        )
        flat_text_gradient = gradient_reducer(flat_text_gradient)
        if (
            not isinstance(flat_recommendation_gradient, torch.Tensor)
            or not isinstance(flat_text_gradient, torch.Tensor)
            or flat_recommendation_gradient.shape
            != flat_text_gradient.shape
        ):
            raise ValueError(
                "Gradient reducer must return matching flat tensors."
            )

    if (
        not torch.isfinite(flat_recommendation_gradient).all()
        or not torch.isfinite(flat_text_gradient).all()
    ):
        raise ValueError(
            "Non-finite gradients encountered during OPSD coordination."
        )

    dot_product = torch.dot(
        flat_text_gradient,
        flat_recommendation_gradient,
    )
    recommendation_norm_sq = flat_recommendation_gradient.square().sum()
    text_norm_sq = flat_text_gradient.square().sum()
    epsilon_tensor = recommendation_norm_sq.new_tensor(epsilon)
    both_nonzero = (
        (recommendation_norm_sq > epsilon_tensor)
        & (text_norm_sq > epsilon_tensor)
    )
    conflict = both_nonzero & (dot_product < 0)
    projection_coefficient = torch.where(
        conflict,
        dot_product / (recommendation_norm_sq + epsilon_tensor),
        torch.zeros_like(dot_product),
    ).detach()
    projected_text_gradient = (
        flat_text_gradient
        - projection_coefficient * flat_recommendation_gradient
    )
    cosine_denominator = torch.sqrt(
        recommendation_norm_sq * text_norm_sq
    ).clamp_min(epsilon_tensor)

    metrics = {
        "opsd_gradient_cosine": (
            dot_product / cosine_denominator
        ).detach(),
        "opsd_gradient_conflict": conflict.float().detach(),
        "opsd_gradient_recommendation_norm": (
            recommendation_norm_sq.sqrt().detach()
        ),
        "opsd_gradient_text_norm": text_norm_sq.sqrt().detach(),
        "opsd_gradient_projection_coefficient": projection_coefficient,
        "opsd_gradient_projected_text_norm": (
            projected_text_gradient.norm().detach()
        ),
        "opsd_gradient_post_projection_dot": torch.dot(
            projected_text_gradient,
            flat_recommendation_gradient,
        ).detach(),
    }
    if mode == "diagnostic_only":
        return additive_loss, metrics

    # alpha is detached.  Adding a zero-valued surrogate changes only the
    # backward direction: grad = g_rec + g_text - alpha * g_rec.
    adjustment = (
        -projection_coefficient.to(
            device=recommendation_loss.device,
            dtype=recommendation_loss.dtype,
        )
        * recommendation_loss
    )
    projected_loss = additive_loss + adjustment - adjustment.detach()
    return projected_loss, metrics


def _recommendation_advantage_gate_weights(
        reward_gap: torch.Tensor,
        enabled: bool,
        mode: str,
        temperature: float,
        power: float,
):
    """Convert detached teacher-minus-student OutcomeGRPO reward into trust weights."""

    if reward_gap.ndim != 1:
        raise ValueError(
            "Recommendation reward gap must contain one value per example."
        )
    mode = _normalize_opsd_teacher_gate_mode(mode)
    temperature = float(temperature)
    power = float(power)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "opsd_teacher_gate_temperature must be finite and positive."
        )
    if not math.isfinite(power) or power <= 0:
        raise ValueError(
            "opsd_teacher_gate_power must be finite and positive."
        )

    gap = reward_gap.detach().float()
    if not bool(enabled):
        weights = torch.ones_like(gap)
    elif mode == "hard":
        # Keep the established strict binary gate exactly: equality is not
        # considered teacher-better.
        weights = (gap > 0).to(gap.dtype)
    else:
        positive_advantage = gap.clamp_min(0.0)
        weights = (
            1.0 - torch.exp(-positive_advantage / temperature)
        ).pow(power)

    metric_zero = gap[:0].sum()
    if weights.numel() == 0:
        weight_mean = metric_zero
        nonzero_rate = metric_zero
        weight_p50 = metric_zero
        weight_p90 = metric_zero
        positive_gap_mean = metric_zero
    else:
        weight_mean = weights.mean()
        nonzero_rate = (weights > 0).float().mean()
        weight_p50 = torch.quantile(weights, 0.5)
        weight_p90 = torch.quantile(weights, 0.9)
        positive = gap > 0
        positive_gap_mean = (
            gap[positive].mean() if positive.any() else metric_zero
        )
    metrics = {
        "opsd_teacher_gate_weight_mean": weight_mean,
        "opsd_teacher_gate_nonzero_rate": nonzero_rate,
        "opsd_teacher_gate_weight_p50": weight_p50,
        "opsd_teacher_gate_weight_p90": weight_p90,
        "opsd_teacher_positive_reward_gap_mean": positive_gap_mean,
    }
    return weights.detach(), {
        key: value.detach() for key, value in metrics.items()
    }


def _weight_completion_token_loss_by_example(
        per_token_loss: torch.Tensor,
        per_example_counts: torch.Tensor,
        per_example_weights: torch.Tensor,
):
    """Apply detached sample weights with token-count-aware normalization."""

    if per_token_loss.ndim != 1:
        raise ValueError("Per-token OPSD loss must be one-dimensional.")
    if per_example_counts.ndim != 1:
        raise ValueError("Per-example token counts must be one-dimensional.")
    if per_example_weights.ndim != 1:
        raise ValueError("Per-example gate weights must be one-dimensional.")
    if per_example_counts.shape[0] != per_example_weights.shape[0]:
        raise ValueError("Gate weights and token-count batch sizes differ.")
    counts = per_example_counts.detach().to(
        device=per_token_loss.device,
        dtype=torch.long,
    )
    if (counts < 0).any() or int(counts.sum().item()) != per_token_loss.numel():
        raise ValueError("Per-example counts do not cover the token losses.")
    weights = per_example_weights.detach().to(
        device=per_token_loss.device,
        dtype=per_token_loss.dtype,
    )
    if not torch.isfinite(weights).all():
        raise ValueError("OPSD teacher gate weights must be finite.")
    if ((weights < 0) | (weights > 1)).any():
        raise ValueError("OPSD teacher gate weights must be in [0, 1].")

    token_weights = torch.repeat_interleave(weights, counts)
    if per_token_loss.numel() > 0:
        weight_sum = token_weights.sum()
        safe_weight_sum = torch.where(
            weight_sum > 0,
            weight_sum,
            torch.ones_like(weight_sum),
        )
        loss = (token_weights * per_token_loss).sum() / safe_weight_sum
        effective_token_weight = token_weights.mean()
    else:
        loss = per_token_loss[:0].sum()
        effective_token_weight = weights[:0].sum()
    return loss, {
        "opsd_teacher_gate_effective_token_weight": (
            effective_token_weight.detach()
        ),
    }


def _normalize_opsd_rank_loss_type(loss_type: str) -> str:
    """Normalize the privileged item-ranking distillation objective."""

    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized not in _OPSD_RANK_LOSS_TYPES:
        choices = ", ".join(sorted(_OPSD_RANK_LOSS_TYPES))
        raise ValueError(
            f"opsd_rank_loss_type must be one of: {choices}."
        )
    return normalized


def _resolve_opsd_teacher_update_mode(
        mode: str,
        teacher_checkpoint: Optional[str],
        token_weight: float,
        rank_weight: float,
        target_weight: float = 0.0,
        target_loss_type: str = "infonce",
        directau_weight: float = 0.0,
) -> str:
    """Resolve automatic teacher lifecycle semantics before Trainer startup."""

    normalized = str(mode).strip().lower()
    if normalized not in _OPSD_TEACHER_UPDATE_MODES:
        choices = ", ".join(sorted(_OPSD_TEACHER_UPDATE_MODES))
        raise ValueError(
            f"opsd_teacher_update_mode must be one of: {choices}."
        )
    has_external_teacher = teacher_checkpoint is not None
    needs_privileged_teacher = (
        token_weight != 0
        or rank_weight != 0
        or directau_weight != 0
        or (
            target_weight != 0
            and _normalize_opsd_target_loss_type(target_loss_type)
            == "privileged_directau"
        )
    )
    if normalized == "auto":
        if has_external_teacher:
            return "fixed"
        if needs_privileged_teacher:
            return "lagged"
        return "live"
    if has_external_teacher and normalized != "fixed":
        raise ValueError(
            "An external opsd_teacher_checkpoint requires "
            "opsd_teacher_update_mode=fixed in the standard OPSD route."
        )
    return normalized


def _copy_trainable_student_parameters(
        student_model,
        teacher_model,
        parameter_names=None,
) -> tuple[tuple[str, ...], float]:
    """Copy the trainable student state into an exact-name snapshot teacher."""

    student_parameters = dict(student_model.named_parameters())
    teacher_parameters = dict(teacher_model.named_parameters())
    if parameter_names is None:
        parameter_names = tuple(
            name
            for name, parameter in student_parameters.items()
            if parameter.requires_grad
        )
    else:
        parameter_names = tuple(parameter_names)
    if not parameter_names:
        raise ValueError(
            "Lagged OPSD teacher found no trainable student parameters to copy."
        )

    missing_student = [
        name for name in parameter_names if name not in student_parameters
    ]
    missing_teacher = [
        name for name in parameter_names if name not in teacher_parameters
    ]
    if missing_student or missing_teacher:
        raise RuntimeError(
            "Lagged OPSD student/teacher parameter names do not match: "
            f"missing_student={missing_student[:3]}, "
            f"missing_teacher={missing_teacher[:3]}."
        )

    squared_update = 0.0
    with torch.no_grad():
        for name in parameter_names:
            source = student_parameters[name].detach()
            destination = teacher_parameters[name]
            if source.shape != destination.shape:
                raise RuntimeError(
                    "Lagged OPSD student/teacher parameter shapes differ for "
                    f"{name}: {tuple(source.shape)} != "
                    f"{tuple(destination.shape)}."
                )
            difference = destination.detach().float() - source.float()
            squared_update += float(difference.square().sum().item())
            destination.copy_(source.to(
                device=destination.device,
                dtype=destination.dtype,
            ))
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    return parameter_names, math.sqrt(squared_update)


def _snapshot_named_parameters(model, parameter_names) -> Dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    missing = [name for name in parameter_names if name not in parameters]
    if missing:
        raise RuntimeError(
            "Cannot snapshot missing OPSD teacher parameters: "
            f"{missing[:3]}."
        )
    return {
        name: parameters[name].detach().cpu().clone()
        for name in parameter_names
    }


def _load_named_parameters(model, state: Dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = [name for name in state if name not in parameters]
    if missing:
        raise RuntimeError(
            "Cannot restore missing OPSD teacher parameters: "
            f"{missing[:3]}."
        )
    with torch.no_grad():
        for name, value in state.items():
            destination = parameters[name]
            if destination.shape != value.shape:
                raise RuntimeError(
                    "Saved OPSD teacher parameter shape differs for "
                    f"{name}: {tuple(value.shape)} != "
                    f"{tuple(destination.shape)}."
                )
            destination.copy_(value.to(
                device=destination.device,
                dtype=destination.dtype,
            ))
    model.requires_grad_(False)
    model.eval()


def _load_tensor_sidecar(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_frozen_opsd_teacher_model(model, checkpoint):
    """Clone ``model`` and optionally activate a stage-1 teacher adapter."""

    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_text = str(checkpoint).strip()
        if not checkpoint_text:
            raise ValueError("opsd_teacher_checkpoint must not be empty.")
        checkpoint_path = Path(checkpoint_text).expanduser()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                "OPSD teacher checkpoint directory does not exist: "
                f"{checkpoint_path}"
            )
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError(
                "OPSD teacher checkpoint is not a PEFT/LoRA adapter; "
                f"missing {adapter_config_path}."
            )
        if not hasattr(model, "load_adapter") or not hasattr(
            model,
            "set_adapter",
        ):
            raise TypeError(
                "opsd_teacher_checkpoint requires a PEFT model with "
                "load_adapter() and set_adapter(); keep use_lora=True."
            )
        if _OPSD_TEACHER_ADAPTER_NAME in getattr(model, "peft_config", {}):
            raise ValueError(
                "The reserved PEFT adapter name "
                f"{_OPSD_TEACHER_ADAPTER_NAME!r} already exists."
            )

    teacher_model = deepcopy(model)
    if checkpoint_path is not None:
        teacher_model.load_adapter(
            str(checkpoint_path),
            adapter_name=_OPSD_TEACHER_ADAPTER_NAME,
            is_trainable=False,
        )
        teacher_model.set_adapter(_OPSD_TEACHER_ADAPTER_NAME)

    # PEFT's set_adapter can mark an adapter trainable, so freeze only after
    # activating the stage-1 adapter.
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    normalized_checkpoint = (
        None if checkpoint_path is None else str(checkpoint_path)
    )
    return teacher_model, normalized_checkpoint


@dataclass
class OPSDTrainingArguments(OutcomeGRPOTrainingArguments):
    """Arguments added to the original OutcomeGRPO training arguments.

    Loss weights default to zero so each objective must be enabled explicitly.
    """

    opsd_target_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the ground-truth target-item InfoNCE loss over "
                "distributed in-batch target items."
            )
        },
    )
    opsd_target_loss_type: str = field(
        default="infonce",
        metadata={
            "help": (
                "Ground-truth recommendation objective: in-batch infonce or "
                "full_softmax_ce over the complete cached item catalog, or "
                "privileged_directau with a training-only teacher."
            )
        },
    )
    opsd_directau_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Independent weight of privileged DirectAU. Unlike the "
                "legacy privileged_directau target-loss type, this additive "
                "objective can run alongside target InfoNCE."
            )
        },
    )
    opsd_privileged_alignment_mode: str = field(
        default="cross_user_item",
        metadata={
            "help": (
                "Privileged DirectAU alignment: reciprocal cross_user_item "
                "or student user_to_user teacher alignment."
            )
        },
    )
    opsd_privileged_alignment_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight of the detached privileged-teacher alignment inside "
                "the privileged DirectAU objective."
            )
        },
    )
    opsd_directau_uniform_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight of student user/item uniformity inside the "
                "privileged DirectAU objective."
            )
        },
    )
    opsd_directau_uniform_temperature: float = field(
        default=2.0,
        metadata={
            "help": (
                "Squared-distance scale used by privileged DirectAU "
                "student uniformity."
            )
        },
    )
    opsd_na_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Independent fixed weight of the LightCCF neighborhood "
                "aggregation loss. This objective is additive and can be "
                "used together with either target recommendation loss."
            )
        },
    )
    opsd_na_temperature: float = field(
        default=0.2,
        metadata={
            "help": "Temperature used only by the LightCCF NA objective."
        },
    )
    opsd_token_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of the token-level OPSD loss."},
    )
    opsd_gradient_coordination_mode: str = field(
        default="sum",
        metadata={
            "help": (
                "Interaction between recommendation-space and token-KD "
                "gradients: sum preserves the original additive objective, "
                "diagnostic_only reports conflicts without changing them, "
                "and rec_preserving_projection removes only the token-KD "
                "component that conflicts with recommendation learning."
            )
        },
    )
    opsd_gradient_coordination_epsilon: float = field(
        default=1e-12,
        metadata={
            "help": (
                "Finite positive stability constant for gradient cosine "
                "and recommendation-preserving projection."
            )
        },
    )
    opsd_token_credit_mode: str = field(
        default="uniform",
        metadata={
            "help": (
                "Token-level OPSD credit policy: uniform preserves the "
                "original mean, while target_item_cosine weights completion "
                "tokens by cosine relevance to the ground-truth item and "
                "teacher_last_layer_attention uses the privileged teacher's "
                "final recommendation-readout attention."
            )
        },
    )
    opsd_token_credit_temperature: float = field(
        default=0.2,
        metadata={
            "help": (
                "Softmax temperature for target-item token relevance."
            )
        },
    )
    opsd_token_credit_mix: float = field(
        default=1.0,
        metadata={
            "help": (
                "Mixture between uniform and recommendation-aware token "
                "credit: 0 is uniform and 1 is fully relevance-aware."
            )
        },
    )
    opsd_rationale_geometry_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the headless rationale-to-target-item relational "
                "geometry objective. Zero disables it without changing "
                "legacy paths."
            )
        },
    )
    opsd_rationale_geometry_loss_type: str = field(
        default="relational_kl",
        metadata={
            "help": (
                "Rationale-to-item relation objective: relational_kl keeps "
                "the original neighbourhood-distribution KL, while "
                "correlation minimizes one minus the Pearson correlation "
                "between off-diagonal rationale and item similarities."
            )
        },
    )
    opsd_rationale_reason_temperature: float = field(
        default=0.2,
        metadata={
            "help": "Temperature for rationale-neighbour similarities."
        },
    )
    opsd_rationale_item_temperature: float = field(
        default=0.2,
        metadata={
            "help": "Temperature for detached target-item similarities."
        },
    )
    opsd_embed_weight: float = field(
        default=0.0,
        metadata={
            "help": "Deprecated alias of opsd_rank_weight for old launch commands."
        },
    )
    opsd_rank_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight of privileged-LLM item-ranking distillation."
        },
    )
    opsd_rank_loss_type: str = field(
        default="distribution_kl",
        metadata={
            "help": (
                "Privileged item objective: distribution_kl preserves the "
                "original score-distribution KD, while geometry_margin "
                "matches target-relative normalized L2 distances."
            )
        },
    )
    opsd_geometry_positive_margin_only: bool = field(
        default=True,
        metadata={
            "help": (
                "For geometry_margin, distil only item pairs for which the "
                "teacher places the target closer than the negative item."
            )
        },
    )
    opsd_rank_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Additional temperature for privileged-LLM item-score KL."
        },
    )
    opsd_rank_topk: int = field(
        default=64,
        metadata={
            "help": (
                "Items taken from both privileged-teacher and student rankings; "
                "the target is always included. Use 0 for the full catalog."
            )
        },
    )
    opsd_teacher_better_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Distil an example only when the privileged teacher has a "
                "higher OutcomeGRPO recommendation reward than the student."
            )
        },
    )
    opsd_teacher_gate_mode: str = field(
        default="hard",
        metadata={
            "help": (
                "Privileged-teacher sample gate: hard preserves the strict "
                "teacher-better mask; advantage_weighted uses normalized "
                "detached positive-advantage weights so their common scale "
                "does not weaken OPSD."
            )
        },
    )
    opsd_teacher_gate_temperature: float = field(
        default=0.1,
        metadata={
            "help": (
                "Positive reward-gap scale for advantage-weighted OPSD."
            )
        },
    )
    opsd_teacher_gate_power: float = field(
        default=1.0,
        metadata={
            "help": "Shape exponent for advantage-weighted OPSD gating."
        },
    )
    opsd_frozen_teacher: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep an independent, frozen snapshot of the initial model "
                "for token-level and/or item-ranking OPSD distillation."
            )
        },
    )
    opsd_teacher_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional stage-1 privileged-teacher PEFT/LoRA checkpoint. "
                "It is loaded only into the independent frozen token/rank/"
                "DirectAU teacher; "
                "init_checkpoint remains the trainable student checkpoint."
            )
        },
    )
    opsd_teacher_update_mode: str = field(
        default="auto",
        metadata={
            "help": (
                "Teacher lifecycle: auto, fixed, lagged, or live. Auto uses "
                "a fixed external checkpoint, a lagged self-teacher when no "
                "checkpoint is supplied, and live mode for target-only runs."
            )
        },
    )
    opsd_teacher_sync_steps: int = field(
        default=128,
        metadata={
            "help": (
                "Optimizer-step interval for lagged teacher synchronization."
            )
        },
    )
    opsd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used by the token-level OPSD loss."},
    )
    opsd_max_token_kl: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional upper bound on each example's mean token KL. "
                "Examples above it are excluded from token OPSD only; "
                "None disables filtering."
            )
        },
    )
    opsd_loss_type: str = field(
        default="kl",
        metadata={"help": "Token divergence: kl, reverse_kl, or jsd."},
    )
    opsd_topk: int = field(
        default=64,
        metadata={
            "help": (
                "Teacher top-k vocabulary size. Use 0 for exact full-vocabulary "
                "distillation."
            )
        },
    )
    opsd_entropy_gate_mode: str = field(
        default="none",
        metadata={
            "help": (
                "Token entropy-gap gate: none or hard. Hard keeps reliable "
                "positions only when H_student - H_teacher exceeds the "
                "configured threshold."
            )
        },
    )
    opsd_entropy_gap_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum detached H_student - H_teacher gap in nats."
        },
    )
    opsd_entropy_gate_temperature: float = field(
        default=0.1,
        metadata={
            "help": (
                "Positive temperature for soft entropy-gap weighting. "
                "Ignored by none/hard modes."
            )
        },
    )
    opsd_entropy_gate_min_teacher_topk_mass: float = field(
        default=0.0,
        metadata={
            "help": (
                "Minimum teacher Top-K mass required to trust entropy gating. "
                "Lower-mass positions receive zero distillation weight."
            )
        },
    )
    opsd_entropy_gate_loss_normalization: str = field(
        default="active_tokens",
        metadata={
            "help": (
                "Entropy-gated token-loss denominator: active_tokens "
                "preserves the original retained-token mean; "
                "candidate_tokens divides the retained loss sum by all "
                "candidate reasoning tokens to isolate token selection "
                "without amplifying KD strength."
            )
        },
    )
    opsd_topk_mass_diagnostics: bool = field(
        default=False,
        metadata={
            "help": (
                "Collect a detached sample of teacher Top-K full-vocabulary "
                "probability mass over reasoning positions and add its "
                "summary to train_results.json."
            )
        },
    )
    opsd_topk_mass_max_positions: int = field(
        default=100000,
        metadata={
            "help": (
                "Maximum number of reasoning-position Top-K masses retained "
                "per process for diagnostics."
            )
        },
    )
    opsd_token_chunk_size: int = field(
        default=64,
        metadata={
            "help": "Number of valid completion positions processed per loss chunk."
        },
    )
    opsd_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum teacher sequence length; defaults to user_input_max_length."
        },
    )
    opsd_privileged_context: str = field(
        default="target",
        metadata={
            "help": (
                "Teacher-only privileged context. With an external teacher, "
                "target/future reuse that teacher's stage-1 prompt format; "
                "target_hard_negatives adds current-student alternatives. "
                "Without an external teacher, target preserves the legacy "
                "strict OPSD prompt."
            )
        },
    )
    opsd_hard_negative_count: int = field(
        default=3,
        metadata={
            "help": (
                "Number of detached current-student full-catalog alternatives "
                "shown only to a target_hard_negatives teacher."
            )
        },
    )
    opsd_hard_negative_exclude_train_positives: bool = field(
        default=True,
        metadata={
            "help": (
                "Exclude every item observed for the user in the training "
                "split when mining hard alternatives."
            )
        },
    )
    opsd_item_bank_refresh_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Full item-bank refresh interval for OPSD. -1 preserves the "
                "legacy partial-update behavior, 0 keeps one consistent "
                "snapshot between evaluations, and a positive value rebuilds "
                "the full bank after that many optimizer steps."
            )
        },
    )
    lightgcn_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Checkpoint produced by the repository's lightgcn.py."
        },
    )
    lightgcn_rank_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of frozen LightGCN candidate-rank KL."},
    )
    lightgcn_topk: int = field(
        default=32,
        metadata={
            "help": (
                "Top items taken from both LightGCN and student rankings; "
                "use 0 for the full catalog."
            )
        },
    )
    lightgcn_teacher_temperature: float = field(
        default=0.2,
        metadata={"help": "Temperature for LightGCN dot-product scores."},
    )
    lightgcn_student_temperature: float = field(
        default=0.2,
        metadata={"help": "Temperature for Recommendation recommendation scores."},
    )
    lightgcn_teacher_better_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply collaborative KL only when LightGCN ranks the train "
                "target above the current student."
            )
        },
    )
    sasrec_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Checkpoint produced by the repository's sasrec.py."
        },
    )
    sasrec_rank_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of frozen SASRec candidate-rank KL."},
    )
    sasrec_topk: int = field(
        default=32,
        metadata={
            "help": (
                "Top items taken from both SASRec and student rankings; "
                "use 0 for the full catalog."
            )
        },
    )
    sasrec_teacher_temperature: float = field(
        default=0.2,
        metadata={"help": "Temperature for SASRec dot-product scores."},
    )
    sasrec_student_temperature: float = field(
        default=0.2,
        metadata={"help": "Temperature for Recommendation recommendation scores."},
    )
    sasrec_teacher_better_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply sequential KL only when SASRec ranks the train "
                "target above the current student."
            )
        },
    )
    sasrec_teacher_gate_mode: str = field(
        default="hard",
        metadata={
            "help": (
                "SASRec sample gate: hard preserves the strict target-rank "
                "comparison; advantage_weighted uses 10x batch-mean "
                "continuous weights from the detached SASRec reward "
                "advantage."
            )
        },
    )
    sasrec_teacher_reward_mode: str = field(
        default="normalized_log_rank",
        metadata={
            "help": (
                "Reward used by the SASRec advantage-weighted gate: "
                "normalized_log_rank uses dense full-catalog rank progress; "
                "candidate_bpr compares target-vs-candidate score margins."
            )
        },
    )
    sasrec_teacher_gate_temperature: float = field(
        default=0.1,
        metadata={
            "help": "Positive reward-gap scale for the SASRec soft gate."
        },
    )
    sasrec_teacher_gate_power: float = field(
        default=1.0,
        metadata={"help": "Shape exponent for the SASRec soft gate."},
    )
    opsd_small_moe_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Independent four-expert MoE loss weight. The pool is fixed "
                "to SASRec, LightGCN, FMLP-Rec, and EASE."
            )
        },
    )
    opsd_small_moe_router_mode: str = field(
        default="soft_mixture",
        metadata={
            "help": (
                "Four-expert MoE routing: soft_mixture combines all positive-"
                "advantage experts; hard_best selects the single best expert."
            )
        },
    )
    opsd_small_moe_advantage_mode: str = field(
        default="target_rank",
        metadata={
            "help": (
                "Four-expert advantage: target_rank (paper default) or "
                "candidate_bpr on the shared candidate union."
            )
        },
    )
    small_moe_sasrec_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "SASRec checkpoint used only by the four-expert MoE."},
    )
    small_moe_lightgcn_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "LightGCN checkpoint used only by the four-expert MoE."},
    )
    small_moe_fmlprec_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "FMLP-Rec checkpoint used only by the four-expert MoE."},
    )
    small_moe_ease_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "EASE checkpoint used only by the four-expert MoE."},
    )
    opsd_small_moe_topk: int = field(
        default=32,
        metadata={
            "help": (
                "Per-model ranking candidates used by the four-expert MoE; "
                "use 0 for the full catalog."
            )
        },
    )
    opsd_small_moe_student_temperature: float = field(
        default=0.2,
        metadata={"help": "Recommendation score temperature inside the MoE KL."},
    )
    opsd_small_moe_sasrec_temperature: float = field(
        default=0.2,
        metadata={"help": "SASRec score temperature inside the MoE."},
    )
    opsd_small_moe_lightgcn_temperature: float = field(
        default=0.2,
        metadata={"help": "LightGCN score temperature inside the MoE."},
    )
    opsd_small_moe_fmlprec_temperature: float = field(
        default=0.2,
        metadata={"help": "FMLP-Rec score temperature inside the MoE."},
    )
    opsd_small_moe_ease_temperature: float = field(
        default=0.2,
        metadata={"help": "EASE score temperature inside the MoE."},
    )
    opsd_small_moe_gate_temperature: float = field(
        default=0.1,
        metadata={"help": "Positive rank-advantage router temperature."},
    )
    opsd_small_moe_outer_temperature: float = field(
        default=0.1,
        metadata={"help": "Outer MoE strength temperature for best advantage."},
    )
    opsd_small_moe_quality_power: float = field(
        default=1.0,
        metadata={"help": "Exponent of each expert's validation-NDCG prior."},
    )


def _scale_loss_gradient(
        loss: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Scale a scalar loss gradient without changing its reported value."""

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Loss gradient scale must be finite and positive.")
    if scale == 1.0:
        return loss
    return loss.detach() + scale * (loss - loss.detach())


def _validate_frozen_teacher_objectives(
        frozen_teacher: bool,
        token_weight: float,
        rank_weight: float,
        privileged_target_enabled: bool = False,
        independent_directau_enabled: bool = False,
) -> None:
    """Require a frozen OPSD teacher to serve at least one objective."""

    if (
        frozen_teacher
        and token_weight == 0
        and rank_weight == 0
        and not privileged_target_enabled
        and not independent_directau_enabled
    ):
        raise ValueError(
            "opsd_frozen_teacher=True requires a non-zero "
            "opsd_token_weight, opsd_rank_weight, or an enabled "
            "privileged_directau objective."
        )


def _select_target_item_embeddings(
        item_hidden: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
) -> torch.Tensor:
    """Select differentiable target embeddings from live candidate items.

    Duplicate candidate rows for the same item are averaged so every live
    positive receives gradient without treating duplicates as negatives.
    """

    if item_hidden.ndim != 2:
        raise ValueError("Live item hidden states must have shape [items, dim].")
    candidate_item_ids = candidate_item_ids.reshape(-1).to(item_hidden.device)
    target_item_ids = target_item_ids.reshape(-1).to(item_hidden.device)
    if candidate_item_ids.shape[0] != item_hidden.shape[0]:
        raise ValueError(
            "Candidate item IDs and live item hidden states have different "
            "lengths."
        )
    positive_mask = target_item_ids.unsqueeze(1).eq(
        candidate_item_ids.unsqueeze(0)
    )
    if not positive_mask.any(dim=1).all():
        raise ValueError(
            "Every privileged DirectAU target must appear among the current "
            "live item prompts."
        )
    weights = positive_mask.to(dtype=item_hidden.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return weights @ item_hidden


def _masked_directau_uniformity(
        representations: torch.Tensor,
        group_ids,
        temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DirectAU uniformity over valid unordered pairs.

    Pairs sharing a supplied group ID are omitted. This excludes repeated
    users from user uniformity and duplicate targets from item uniformity.
    """

    if representations.ndim != 2:
        raise ValueError("DirectAU representations must have shape [batch, dim].")
    batch_size = int(representations.shape[0])
    if batch_size < 2:
        zero = representations.sum() * 0.0
        return zero, representations.new_zeros((), dtype=torch.float32)

    pair_rows, pair_columns = torch.triu_indices(
        batch_size,
        batch_size,
        offset=1,
        device=representations.device,
    )
    valid_pairs = torch.ones(
        pair_rows.shape[0],
        dtype=torch.bool,
        device=representations.device,
    )
    if group_ids is not None:
        if isinstance(group_ids, torch.Tensor):
            group_ids = group_ids.to(representations.device).reshape(-1)
        else:
            raw_group_ids = list(group_ids)
            encoded_groups = {}
            encoded_group_ids = []
            for raw_group_id in raw_group_ids:
                if raw_group_id not in encoded_groups:
                    encoded_groups[raw_group_id] = len(encoded_groups)
                encoded_group_ids.append(encoded_groups[raw_group_id])
            group_ids = torch.tensor(
                encoded_group_ids,
                dtype=torch.long,
                device=representations.device,
            )
        if group_ids.shape[0] != batch_size:
            raise ValueError(
                "DirectAU group IDs and representations have different "
                "batch sizes."
            )
        valid_pairs &= group_ids[pair_rows].ne(group_ids[pair_columns])

    pair_rows = pair_rows[valid_pairs]
    pair_columns = pair_columns[valid_pairs]
    pair_count = int(pair_rows.numel())
    if pair_count == 0:
        zero = representations.sum() * 0.0
        return zero, representations.new_zeros((), dtype=torch.float32)

    squared_distances = (
        representations[pair_rows] - representations[pair_columns]
    ).square().sum(dim=-1)
    scaled_distances = -float(temperature) * squared_distances
    uniformity = torch.logsumexp(scaled_distances, dim=0) - math.log(pair_count)
    return uniformity, representations.new_tensor(
        float(pair_count),
        dtype=torch.float32,
    )


def _privileged_directau_loss(
        student_user_hidden: torch.Tensor,
        student_target_item_hidden: torch.Tensor,
        teacher_user_hidden: torch.Tensor,
        teacher_target_item_hidden: torch.Tensor,
        target_item_ids: torch.Tensor,
        user_ids=None,
        alignment_mode: str = "cross_user_item",
        privileged_alignment_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        uniformity_temperature: float = 2.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Align privileged teacher views while spreading student embeddings.

    The deployable student always retains direct user-target alignment. The
    selected privileged term either uses reciprocal cross-model user-item
    alignment or same-modality student-user to teacher-user alignment.
    Teacher representations are detached inside this helper by construction.
    """

    tensors = (
        student_user_hidden,
        student_target_item_hidden,
        teacher_user_hidden,
        teacher_target_item_hidden,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError(
            "Privileged DirectAU user/item representations must be matrices."
        )
    if len({tensor.shape for tensor in tensors}) != 1:
        raise ValueError(
            "Privileged DirectAU teacher/student representation shapes differ."
        )
    batch_size = int(student_user_hidden.shape[0])
    target_item_ids = target_item_ids.reshape(-1).to(
        student_user_hidden.device
    )
    if target_item_ids.shape[0] != batch_size:
        raise ValueError(
            "Privileged DirectAU target IDs and representations have "
            "different batch sizes."
        )
    alignment_mode = _normalize_opsd_privileged_alignment_mode(
        alignment_mode
    )
    privileged_alignment_weight = float(privileged_alignment_weight)
    uniformity_weight = float(uniformity_weight)
    uniformity_temperature = float(uniformity_temperature)
    for name, value in (
        ("privileged_alignment_weight", privileged_alignment_weight),
        ("uniformity_weight", uniformity_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if (
        not math.isfinite(uniformity_temperature)
        or uniformity_temperature <= 0
    ):
        raise ValueError(
            "uniformity_temperature must be finite and positive."
        )

    student_users = F.normalize(
        student_user_hidden.float(), p=2, dim=-1, eps=1e-12
    )
    student_items = F.normalize(
        student_target_item_hidden.float(), p=2, dim=-1, eps=1e-12
    )
    teacher_users = F.normalize(
        teacher_user_hidden.detach().to(
            device=student_users.device,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    teacher_items = F.normalize(
        teacher_target_item_hidden.detach().to(
            device=student_users.device,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
        eps=1e-12,
    )

    self_alignment = (
        student_users - student_items
    ).square().sum(dim=-1).mean()
    student_user_teacher_item_alignment = (
        student_users - teacher_items
    ).square().sum(dim=-1).mean()
    student_item_teacher_user_alignment = (
        student_items - teacher_users
    ).square().sum(dim=-1).mean()
    student_user_teacher_user_alignment = (
        student_users - teacher_users
    ).square().sum(dim=-1).mean()
    if alignment_mode == "cross_user_item":
        privileged_alignment = 0.5 * (
            student_user_teacher_item_alignment
            + student_item_teacher_user_alignment
        )
    else:
        privileged_alignment = student_user_teacher_user_alignment

    user_uniformity, user_pair_count = _masked_directau_uniformity(
        student_users,
        group_ids=user_ids,
        temperature=uniformity_temperature,
    )
    item_uniformity, item_pair_count = _masked_directau_uniformity(
        student_items,
        group_ids=target_item_ids,
        temperature=uniformity_temperature,
    )
    uniformity = 0.5 * (user_uniformity + item_uniformity)
    loss = (
        self_alignment
        + privileged_alignment_weight * privileged_alignment
        + uniformity_weight * uniformity
    )
    return loss, {
        "opsd_directau_self_alignment": self_alignment.detach(),
        "opsd_directau_privileged_alignment": privileged_alignment.detach(),
        "opsd_directau_user_uniformity": user_uniformity.detach(),
        "opsd_directau_item_uniformity": item_uniformity.detach(),
        "opsd_directau_user_pair_count": user_pair_count.detach(),
        "opsd_directau_item_pair_count": item_pair_count.detach(),
        "opsd_directau_student_user_teacher_item_alignment": (
            student_user_teacher_item_alignment.detach()
        ),
        "opsd_directau_student_item_teacher_user_alignment": (
            student_item_teacher_user_alignment.detach()
        ),
        "opsd_directau_student_user_teacher_user_alignment": (
            student_user_teacher_user_alignment.detach()
        ),
    }


def _target_item_infonce_loss(
        student_scores: torch.Tensor,
        anchor_target_item_ids: torch.LongTensor,
        candidate_item_ids: torch.LongTensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Multi-positive target InfoNCE over distributed in-batch items.

    ``student_scores`` already contains Recommendation's configured similarity
    temperature. Every candidate with the anchor's ground-truth item ID is a
    positive, so duplicate targets within or across ranks are not treated as
    false negatives. Unlike the OPSD losses, this target objective is never
    filtered by the teacher-better gate.
    """

    if student_scores.ndim != 2:
        raise ValueError("Target InfoNCE scores must have shape [users, items].")
    if student_scores.shape[1] == 0:
        raise ValueError("Target InfoNCE requires at least one candidate item.")

    anchor_target_item_ids = anchor_target_item_ids.reshape(-1).to(
        student_scores.device
    )
    candidate_item_ids = candidate_item_ids.reshape(-1).to(
        student_scores.device
    )
    if anchor_target_item_ids.shape[0] != student_scores.shape[0]:
        raise ValueError(
            "Target item IDs do not match the InfoNCE score row count."
        )
    if candidate_item_ids.shape[0] != student_scores.shape[1]:
        raise ValueError(
            "Candidate item IDs do not match the InfoNCE score column count."
        )

    positive_mask = anchor_target_item_ids.unsqueeze(1).eq(
        candidate_item_ids.unsqueeze(0)
    )
    if not positive_mask.any(dim=1).all():
        raise ValueError(
            "Every target InfoNCE anchor must match at least one candidate item."
        )

    float_scores = student_scores.float()
    positive_scores = float_scores.masked_fill(~positive_mask, -torch.inf)
    log_denominator = torch.logsumexp(float_scores, dim=-1)
    log_numerator = torch.logsumexp(positive_scores, dim=-1)
    loss = (log_denominator - log_numerator).mean()

    with torch.no_grad():
        detached_scores = float_scores.detach()
        log_probs = F.log_softmax(detached_scores, dim=-1)
        positive_probability = torch.logsumexp(
            log_probs.masked_fill(~positive_mask, -torch.inf),
            dim=-1,
        ).exp()
        best_positive_score = detached_scores.masked_fill(
            ~positive_mask,
            -torch.inf,
        ).max(dim=-1).values
        negative_mask = ~positive_mask

        # Rank the best positive only against negatives. Other columns with
        # the same item ID must neither worsen its rank nor invalidate top-1.
        greater_negative_count = (
            (detached_scores > best_positive_score.unsqueeze(1))
            & negative_mask
        ).sum(dim=-1)
        tied_negative_count = (
            torch.isclose(
                detached_scores,
                best_positive_score.unsqueeze(1),
                rtol=1e-5,
                atol=1e-7,
            )
            & negative_mask
        ).sum(dim=-1)
        target_rank = (
            greater_negative_count.float()
            + 1.0
            + tied_negative_count.float() / 2.0
        )
        best_negative_score = detached_scores.masked_fill(
            ~negative_mask,
            -torch.inf,
        ).max(dim=-1).values
        positive_top1 = best_positive_score > best_negative_score
        positive_count = positive_mask.sum(dim=-1)
        score_span = (
            detached_scores.max(dim=-1).values
            - detached_scores.min(dim=-1).values
        )

    metrics = {
        "opsd_target_infonce_probability": positive_probability.mean(),
        "opsd_target_infonce_rank": target_rank.mean(),
        "opsd_target_infonce_top1": positive_top1.float().mean(),
        "opsd_target_infonce_candidate_count": float_scores.new_tensor(
            float(float_scores.shape[1])
        ),
        "opsd_target_infonce_positive_count": positive_count.float().mean(),
        "opsd_target_infonce_duplicate_positive_rate": (
            positive_count > 1
        ).float().mean(),
        "opsd_target_infonce_score_span": score_span.mean(),
    }
    return loss, metrics


def _lightccf_neighborhood_aggregation_loss(
        in_batch_item_scores: torch.Tensor,
        paired_user_hidden: torch.Tensor,
        paired_target_item_ids: torch.LongTensor,
        candidate_item_ids: torch.LongTensor,
        similarity_temperature: float,
        na_temperature: float,
        paired_group_ids: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """LightCCF NA over aligned user--positive-item interaction pairs.

    ``in_batch_item_scores`` contains normalized-dot-product logits already
    divided by Recommendation's recommendation temperature.  LightCCF uses its own
    temperature, so the item term is rescaled before it is combined with the
    cosine user--user term.  Candidate item columns are selected by ID because
    pure OPSD mini-batches slice rollout users while retaining macro-batch item
    prompts.
    """

    if in_batch_item_scores.ndim != 2:
        raise ValueError("LightCCF NA item scores must have shape [users, items].")
    if paired_user_hidden.ndim != 2:
        raise ValueError("LightCCF NA user hidden states must be two-dimensional.")
    if in_batch_item_scores.shape[0] != paired_user_hidden.shape[0]:
        raise ValueError(
            "LightCCF NA score rows must match the paired user count."
        )
    if not math.isfinite(similarity_temperature) or similarity_temperature <= 0:
        raise ValueError("similarity_temperature must be finite and positive.")
    if not math.isfinite(na_temperature) or na_temperature <= 0:
        raise ValueError("opsd_na_temperature must be finite and positive.")

    paired_target_item_ids = paired_target_item_ids.reshape(-1).to(
        in_batch_item_scores.device
    )
    candidate_item_ids = candidate_item_ids.reshape(-1).to(
        in_batch_item_scores.device
    )
    pair_count = paired_user_hidden.shape[0]
    if paired_target_item_ids.shape[0] != pair_count:
        raise ValueError(
            "LightCCF NA target IDs must match the paired user count."
        )
    if candidate_item_ids.shape[0] != in_batch_item_scores.shape[1]:
        raise ValueError(
            "LightCCF NA candidate IDs must match the item-score columns."
        )

    pair_item_matches = paired_target_item_ids.unsqueeze(1).eq(
        candidate_item_ids.unsqueeze(0)
    )
    if not pair_item_matches.any(dim=1).all():
        raise ValueError(
            "Every LightCCF NA pair must find its target among candidate items."
        )
    pair_item_columns = pair_item_matches.to(torch.int64).argmax(dim=1)

    item_logits = (
        in_batch_item_scores.float()
        * float(similarity_temperature)
        / float(na_temperature)
    )
    pair_item_logits = item_logits.index_select(1, pair_item_columns)
    normalized_users = F.normalize(paired_user_hidden.float(), p=2, dim=-1)
    pair_user_logits = torch.matmul(
        normalized_users,
        normalized_users.transpose(0, 1),
    ) / float(na_temperature)
    pair_logits = pair_item_logits + pair_user_logits
    same_group_non_diagonal = torch.zeros(
        (pair_count, pair_count),
        dtype=torch.bool,
        device=pair_logits.device,
    )
    if paired_group_ids is not None:
        paired_group_ids = paired_group_ids.reshape(-1).to(pair_logits.device)
        if paired_group_ids.shape[0] != pair_count:
            raise ValueError(
                "LightCCF NA group IDs must match the paired user count."
            )
        same_group = paired_group_ids.unsqueeze(1).eq(
            paired_group_ids.unsqueeze(0)
        )
        same_group_non_diagonal = same_group & ~torch.eye(
            pair_count,
            dtype=torch.bool,
            device=pair_logits.device,
        )
        pair_logits = pair_logits.masked_fill(
            same_group_non_diagonal,
            -torch.inf,
        )
    positive_item_logits = pair_item_logits.diagonal()
    per_pair_loss = torch.logsumexp(pair_logits, dim=-1) - positive_item_logits
    loss = per_pair_loss.mean()

    with torch.no_grad():
        detached_pair_logits = pair_logits.detach()
        detached_positive_logits = positive_item_logits.detach()
        positive_ratio = torch.exp(
            detached_positive_logits
            - torch.logsumexp(detached_pair_logits, dim=-1)
        )
        pair_score_span = (
            detached_pair_logits.max(dim=-1).values
            - detached_pair_logits.min(dim=-1).values
        )

    metrics = {
        "opsd_na_positive_ratio": positive_ratio.mean(),
        "opsd_na_candidate_pair_count": item_logits.new_tensor(
            float(pair_count)
        ),
        "opsd_na_effective_candidate_pair_count": (
            (~same_group_non_diagonal).sum(dim=-1).float().mean()
        ),
        "opsd_na_masked_same_group_count": (
            same_group_non_diagonal.sum(dim=-1).float().mean()
        ),
        "opsd_na_pair_score_span": pair_score_span.mean(),
        "opsd_na_temperature": item_logits.new_tensor(float(na_temperature)),
    }
    return loss, metrics


def _target_item_full_softmax_ce_loss(
        student_scores: torch.Tensor,
        target_item_ids: torch.LongTensor,
        history_item_ids: torch.LongTensor,
        live_target_scores: Optional[torch.Tensor] = None,
        padding_item_id: int = 0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full-catalog target CE with an optional live positive score.

    ``student_scores`` may come from a detached item memory bank. Replacing
    each target column with ``live_target_scores`` preserves gradients through
    the current positive item encoder without retaining a graph for every
    catalog item.
    """

    if student_scores.ndim != 2:
        raise ValueError(
            "Full Softmax CE scores must have shape [users, items]."
        )
    batch_size, num_items = student_scores.shape
    if num_items <= 1:
        raise ValueError("Full Softmax CE requires at least two catalog rows.")
    target_item_ids = target_item_ids.reshape(-1).to(student_scores.device)
    if target_item_ids.shape[0] != batch_size:
        raise ValueError(
            "Full Softmax CE target count does not match score rows."
        )
    history_item_ids = history_item_ids.to(
        device=student_scores.device,
        dtype=torch.long,
    )
    if history_item_ids.ndim != 2 or history_item_ids.shape[0] != batch_size:
        raise ValueError(
            "Full Softmax CE history IDs must have shape [users, history]."
        )
    if not 0 <= int(padding_item_id) < num_items:
        raise ValueError("padding_item_id is outside the item catalog.")
    if (
        (target_item_ids < 0)
        | (target_item_ids >= num_items)
        | target_item_ids.eq(int(padding_item_id))
    ).any():
        raise ValueError(
            "A Full Softmax CE target is padding or outside the item catalog."
        )

    float_scores = student_scores.float()
    if live_target_scores is not None:
        live_target_scores = live_target_scores.reshape(-1).to(
            device=student_scores.device,
            dtype=float_scores.dtype,
        )
        if live_target_scores.shape[0] != batch_size:
            raise ValueError(
                "Live Full Softmax target scores do not match score rows."
            )
        float_scores = float_scores.scatter(
            1,
            target_item_ids.unsqueeze(1),
            live_target_scores.unsqueeze(1),
        )

    available_item_mask = torch.ones(
        (batch_size, num_items),
        dtype=torch.bool,
        device=student_scores.device,
    )
    available_item_mask[:, int(padding_item_id)] = False
    valid_history = (
        (history_item_ids >= 0)
        & (history_item_ids < num_items)
        & history_item_ids.ne(int(padding_item_id))
    )
    if valid_history.any():
        history_rows = torch.arange(
            batch_size,
            device=student_scores.device,
        ).unsqueeze(1).expand_as(history_item_ids)
        available_item_mask[
            history_rows[valid_history],
            history_item_ids[valid_history],
        ] = False
    # Keep the supervised target available even if malformed input repeats it
    # in the history. Prepared Recommendation datasets normally never do so.
    available_item_mask.scatter_(
        1,
        target_item_ids.unsqueeze(1),
        True,
    )
    masked_scores = float_scores.masked_fill(
        ~available_item_mask,
        torch.finfo(float_scores.dtype).min,
    )
    loss = F.cross_entropy(masked_scores, target_item_ids)

    with torch.no_grad():
        detached_scores = masked_scores.detach()
        target_scores = detached_scores.gather(
            1, target_item_ids.unsqueeze(1)
        ).squeeze(1)
        negative_mask = available_item_mask.clone()
        negative_mask.scatter_(1, target_item_ids.unsqueeze(1), False)
        greater_negative_count = (
            (detached_scores > target_scores.unsqueeze(1)) & negative_mask
        ).sum(dim=-1)
        tied_negative_count = (
            torch.isclose(
                detached_scores,
                target_scores.unsqueeze(1),
                rtol=1e-5,
                atol=1e-7,
            )
            & negative_mask
        ).sum(dim=-1)
        target_rank = (
            greater_negative_count.float()
            + 1.0
            + tied_negative_count.float() / 2.0
        )
        best_negative_score = detached_scores.masked_fill(
            ~negative_mask,
            -torch.inf,
        ).max(dim=-1).values
        target_probability = F.softmax(
            detached_scores,
            dim=-1,
        ).gather(1, target_item_ids.unsqueeze(1)).squeeze(1)
        available_max = detached_scores.masked_fill(
            ~available_item_mask,
            -torch.inf,
        ).max(dim=-1).values
        available_min = detached_scores.masked_fill(
            ~available_item_mask,
            torch.inf,
        ).min(dim=-1).values
        masked_history_count = (
            (~available_item_mask).sum(dim=-1).float() - 1.0
        ).clamp_min(0.0)

    metrics = {
        "opsd_target_full_softmax_probability": target_probability.mean(),
        "opsd_target_full_softmax_rank": target_rank.mean(),
        "opsd_target_full_softmax_top1": (
            target_scores > best_negative_score
        ).float().mean(),
        "opsd_target_full_softmax_candidate_count": (
            available_item_mask.sum(dim=-1).float().mean()
        ),
        "opsd_target_full_softmax_masked_history_count": (
            masked_history_count.mean()
        ),
        "opsd_target_full_softmax_score_span": (
            available_max - available_min
        ).mean(),
        "opsd_target_full_softmax_live_positive": float_scores.new_tensor(
            float(live_target_scores is not None)
        ),
    }
    return loss, metrics


def _masked_token_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        temperature: float = 1.0,
        loss_type: str = "jsd",
        chunk_size: int = 8,
) -> torch.Tensor:
    """Compute strict OPSD divergence on the shared student completion.

    ``student_labels`` and ``teacher_labels`` contain ``-100`` outside the
    completion.  Causal logits at position ``j - 1`` predict the token at
    label position ``j``, so the function shifts both masks before selecting
    logits.  Prompt lengths may differ because the masks are built
    independently for student and teacher inputs.
    """

    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("OPSD logits must have shape (batch, sequence, vocab).")
    if student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError("Student and teacher batches must have the same size.")
    if student_labels.ndim != 2 or teacher_labels.ndim != 2:
        raise ValueError("OPSD labels must have shape (batch, sequence).")
    if student_logits.shape[1] != student_labels.shape[1]:
        raise ValueError("Student logits and labels have different sequence lengths.")
    if teacher_logits.shape[1] != teacher_labels.shape[1]:
        raise ValueError("Teacher logits and labels have different sequence lengths.")
    if student_logits.shape[2] != teacher_logits.shape[2]:
        raise ValueError("Student and teacher vocab dimensions must match.")
    if temperature <= 0:
        raise ValueError("opsd_temperature must be positive.")
    if chunk_size <= 0:
        raise ValueError("opsd_token_chunk_size must be positive.")

    # logits[:, j - 1] predicts labels[:, j].  The labels generated by the
    # existing RecommendationDataCollator identify exactly the student completion.
    student_shifted_logits = student_logits[:, :-1, :]
    teacher_shifted_logits = teacher_logits[:, :-1, :]
    student_mask = student_labels[:, 1:] != -100
    teacher_mask = teacher_labels[:, 1:] != -100

    selected_student = []
    selected_teacher = []
    for row in range(student_logits.shape[0]):
        student_row_mask = student_mask[row]
        teacher_row_mask = teacher_mask[row]
        student_targets = student_labels[row, 1:][student_row_mask]
        teacher_targets = teacher_labels[row, 1:][teacher_row_mask]

        if student_targets.numel() != teacher_targets.numel():
            raise ValueError(
                "Student and teacher completion lengths differ after batching. "
                "The teacher must consume the same student completion tokens."
            )
        if not torch.equal(student_targets, teacher_targets):
            raise ValueError(
                "Student and teacher completion token ids are not identical."
            )
        if student_targets.numel() > 0:
            selected_student.append(student_shifted_logits[row][student_row_mask])
            selected_teacher.append(teacher_shifted_logits[row][teacher_row_mask])

    if not selected_student:
        # Keep the result connected to the student graph while contributing
        # zero.  This makes an empty generated completion safe to handle.
        return student_logits.sum() * 0.0

    student_selected = torch.cat(selected_student, dim=0)
    teacher_selected = torch.cat(selected_teacher, dim=0).detach()

    divergence = loss_type.lower()
    if divergence not in {"jsd", "kl", "reverse_kl"}:
        raise ValueError(
            f"Unsupported opsd_loss_type={loss_type!r}; use 'kl', "
            "'reverse_kl', or 'jsd'."
        )

    total = student_selected.new_zeros((), dtype=torch.float32)
    total_count = student_selected.new_tensor(0.0, dtype=torch.float32)
    log_two = math.log(2.0)

    # Chunking avoids materialising all float32 log-probabilities at once.
    # The model still produces the selected logits, but the temporary loss
    # tensors are bounded by opsd_token_chunk_size.
    for start in range(0, student_selected.shape[0], chunk_size):
        end = min(start + chunk_size, student_selected.shape[0])
        student_chunk = student_selected[start:end].float() / temperature
        teacher_chunk = teacher_selected[start:end].float() / temperature

        student_log_probs = F.log_softmax(student_chunk, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)

        if divergence == "kl":
            # KL(p_teacher || p_student), with teacher as the detached target.
            chunk_loss = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        elif divergence == "reverse_kl":
            # KL(p_student || p_teacher), with teacher detached above.
            student_probs = student_log_probs.exp()
            chunk_loss = (
                student_probs
                * (student_log_probs - teacher_log_probs)
            ).sum(dim=-1)
        else:
            # JSD(p_student, p_teacher).
            midpoint_log_probs = torch.logaddexp(
                student_log_probs, teacher_log_probs) - log_two
            student_to_midpoint = F.kl_div(
                midpoint_log_probs,
                student_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            teacher_to_midpoint = F.kl_div(
                midpoint_log_probs,
                teacher_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            chunk_loss = 0.5 * (student_to_midpoint + teacher_to_midpoint)

        total = total + chunk_loss.float().sum()
        total_count = total_count + float(chunk_loss.numel())

    return total / total_count * (temperature ** 2)


def _recommendation_reward_diagnostics(
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        history_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        reward_ndcg_k: int = 1000,
        teacher_better_only: bool = True,
        return_per_example: bool = False,
):
    """Compare teacher/student recommendation quality on the full item bank."""

    if student_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError("Recommendation scores must have shape (batch, items).")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("Student and teacher recommendation scores must match.")
    if history_item_ids.ndim != 2:
        raise ValueError("History item ids must have shape (batch, history).")
    if history_item_ids.shape[0] != student_scores.shape[0]:
        raise ValueError("History and score batch sizes differ.")
    if target_item_ids.ndim != 1:
        target_item_ids = target_item_ids.reshape(-1)
    if target_item_ids.shape[0] != student_scores.shape[0]:
        raise ValueError("Target item ids and score batch sizes differ.")
    if reward_ndcg_k <= 0:
        raise ValueError("reward_ndcg_k must be positive.")
    if teacher_better_only:
        raise NotImplementedError

    batch_size, num_items = student_scores.shape
    if num_items <= 1:
        raise ValueError("Recommendation distillation needs at least two items.")
    if ((target_item_ids < 0) | (target_item_ids >= num_items)).any():
        raise ValueError("A target item id is outside the item embedding bank.")

    history_mask = torch.zeros(
        (batch_size, num_items),
        dtype=torch.bool,
        device=student_scores.device,
    )
    history_item_ids = history_item_ids.to(student_scores.device)
    valid_history = (
        (history_item_ids >= 0) & (history_item_ids < num_items)
    )
    if valid_history.any():
        rows = torch.arange(
            batch_size, device=student_scores.device
        ).unsqueeze(1).expand_as(history_item_ids)
        history_mask[
            rows[valid_history], history_item_ids[valid_history]
        ] = True
    # The preprocessed item with id 0 is the padding item, not a candidate.
    history_mask[:, 0] = True

    target_item_ids = target_item_ids.to(student_scores.device)
    if history_mask.gather(1, target_item_ids.unsqueeze(1)).any():
        raise ValueError("A target item is masked as a historical/padding item.")
    if history_mask.all(dim=-1).any():
        raise ValueError("An OPSD example has no recommendation candidates.")

    with torch.no_grad():
        student_masked = student_scores.detach().float().masked_fill(
            history_mask, -torch.inf
        )
        teacher_masked = teacher_scores.detach().float().masked_fill(
            history_mask, -torch.inf
        )

        student_log_probs = F.log_softmax(student_masked, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_masked, dim=-1)
        target_index = target_item_ids.unsqueeze(1)
        student_target_prob_per_example = student_log_probs.gather(
            1, target_index
        ).exp().squeeze(1)
        teacher_target_prob_per_example = teacher_log_probs.gather(
            1, target_index
        ).exp().squeeze(1)

        student_target_score = student_masked.gather(1, target_index)
        teacher_target_score = teacher_masked.gather(1, target_index)
        student_rank = (
            (student_masked > student_target_score).sum(dim=-1) + 1
        )
        teacher_rank = (
            (teacher_masked > teacher_target_score).sum(dim=-1) + 1
        )

        def _ndcg_at(rank: torch.Tensor, cutoff: int) -> torch.Tensor:
            discounted_gain = 1.0 / torch.log2(rank.float() + 1.0)
            return torch.where(
                rank <= cutoff,
                discounted_gain,
                torch.zeros_like(discounted_gain),
            )

        student_ndcg_reward = _ndcg_at(student_rank, reward_ndcg_k)
        teacher_ndcg_reward = _ndcg_at(teacher_rank, reward_ndcg_k)
        student_ndcg_10 = _ndcg_at(student_rank, 10)
        teacher_ndcg_10 = _ndcg_at(teacher_rank, 10)

        # student_reward = (
        #     (1.0 - reward_softmax_weight) * student_ndcg_reward
        #     + reward_softmax_weight * student_target_prob_per_example
        # )
        # teacher_reward = (
        #     (1.0 - reward_softmax_weight) * teacher_ndcg_reward
        #     + reward_softmax_weight * teacher_target_prob_per_example
        # )
        distill_mask = torch.ones_like(student_rank, dtype=torch.bool)
        teacher_gate_signal = torch.zeros_like(student_ndcg_reward)

    metrics = {
        "opsd_student_target_prob": student_target_prob_per_example.mean(),
        "opsd_teacher_target_prob": teacher_target_prob_per_example.mean(),
        "opsd_teacher_better_rate": distill_mask.float().mean(),
        "opsd_target_prob_gap": (
            teacher_target_prob_per_example
            - student_target_prob_per_example
        ).mean(),
        "opsd_student_target_top1": (student_rank == 1).float().mean(),
        "opsd_teacher_target_top1": (teacher_rank == 1).float().mean(),
        "opsd_student_target_rank": student_rank.float().mean(),
        "opsd_teacher_target_rank": teacher_rank.float().mean(),
        "opsd_student_ndcg@10": student_ndcg_10.mean(),
        "opsd_teacher_ndcg@10": teacher_ndcg_10.mean(),
        f"opsd_student_ndcg@{reward_ndcg_k}": student_ndcg_reward.mean(),
        f"opsd_teacher_ndcg@{reward_ndcg_k}": teacher_ndcg_reward.mean(),
    }
    if return_per_example:
        per_example = {
            "student_target_prob": student_target_prob_per_example,
            "teacher_target_prob": teacher_target_prob_per_example,
            "target_prob_gap": (
                teacher_target_prob_per_example
                - student_target_prob_per_example
            ),
            "student_target_top1": (student_rank == 1),
            "teacher_target_top1": (teacher_rank == 1),
            "student_target_rank": student_rank,
            "teacher_target_rank": teacher_rank,
            "student_ndcg@10": student_ndcg_10,
            "teacher_ndcg@10": teacher_ndcg_10,
            f"student_ndcg@{reward_ndcg_k}": student_ndcg_reward,
            f"teacher_ndcg@{reward_ndcg_k}": teacher_ndcg_reward,
            "teacher_gate_signal": teacher_gate_signal,
            "teacher_better": distill_mask,
        }
        return metrics, distill_mask, per_example
    return metrics, distill_mask


def _build_privileged_item_candidate_mask(
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        history_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        topk: int,
):
    """Build the shared OPSD item set without changing the legacy protocol."""

    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError(
            "Student/privileged-teacher scores must share shape [batch, items]."
        )
    if history_item_ids.ndim != 2:
        raise ValueError("History item ids must have shape [batch, history].")
    batch_size, num_items = student_scores.shape
    if history_item_ids.shape[0] != batch_size:
        raise ValueError("History and item-score batch sizes differ.")
    if num_items <= 1:
        raise ValueError("Item-ranking distillation needs at least two items.")
    if topk < 0:
        raise ValueError(
            "opsd_rank_topk must be non-negative; use 0 for full catalog."
        )

    target_item_ids = target_item_ids.reshape(-1).to(
        device=student_scores.device,
        dtype=torch.long,
    )
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("Target item ids and item-score batch sizes differ.")
    if ((target_item_ids <= 0) | (target_item_ids >= num_items)).any():
        raise ValueError("A target item id is outside the real item catalog.")

    history_item_ids = history_item_ids.to(
        device=student_scores.device,
        dtype=torch.long,
    )
    available = torch.ones(
        (batch_size, num_items),
        dtype=torch.bool,
        device=student_scores.device,
    )
    available[:, 0] = False
    valid_history = (
        (history_item_ids > 0) & (history_item_ids < num_items)
    )
    if valid_history.any():
        rows = torch.arange(
            batch_size, device=student_scores.device
        ).unsqueeze(1).expand_as(history_item_ids)
        available[rows[valid_history], history_item_ids[valid_history]] = False

    target_index = target_item_ids.unsqueeze(1)
    if (~available.gather(1, target_index)).any():
        raise ValueError("A target item is padding or already present in history.")
    if (~available).all(dim=-1).any():
        raise ValueError("An OPSD example has no recommendation candidates.")

    if topk == 0:
        candidate_mask = available
    else:
        k = min(topk, num_items - 1)
        student_for_selection = student_scores.detach().float().masked_fill(
            ~available, -torch.inf
        )
        teacher_for_selection = teacher_scores.detach().to(
            student_scores.device
        ).float().masked_fill(~available, -torch.inf)
        teacher_topk = torch.topk(
            teacher_for_selection, k=k, dim=-1
        ).indices
        student_topk = torch.topk(
            student_for_selection, k=k, dim=-1
        ).indices
        candidate_mask = torch.zeros_like(available)
        candidate_mask.scatter_(1, teacher_topk, True)
        candidate_mask.scatter_(1, student_topk, True)
        candidate_mask.scatter_(1, target_index, True)
        candidate_mask &= available

    return candidate_mask, target_item_ids


def _privileged_item_geometry_distillation_loss(
        student_user_hidden: torch.Tensor,
        teacher_user_hidden: torch.Tensor,
        student_item_hs: torch.Tensor,
        teacher_item_hs: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        history_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        distill_mask: torch.Tensor,
        distill_weights=None,
        topk: int = 64,
        positive_margin_only: bool = True,
):
    """Match target-relative user-item distances in each model's own space.

    User and item vectors are L2-normalized independently for the student and
    privileged teacher.  The teacher supplies detached relative distances;
    gradients flow only through the student user representation.  Candidate
    IDs follow the legacy item-KL protocol exactly.
    """

    if student_user_hidden.ndim != 2 or teacher_user_hidden.ndim != 2:
        raise ValueError("Student and teacher user hidden states must be rank 2.")
    if student_item_hs.ndim != 2 or teacher_item_hs.ndim != 2:
        raise ValueError("Student and teacher item banks must be rank 2.")
    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError(
            "Student/privileged-teacher scores must share shape [batch, items]."
        )
    batch_size, num_items = student_scores.shape
    if student_user_hidden.shape[0] != batch_size:
        raise ValueError("Student user and item-score batch sizes differ.")
    if teacher_user_hidden.shape[0] != batch_size:
        raise ValueError("Teacher user and item-score batch sizes differ.")
    if student_item_hs.shape[0] != num_items:
        raise ValueError("Student item bank and score catalog sizes differ.")
    if teacher_item_hs.shape[0] != num_items:
        raise ValueError("Teacher item bank and score catalog sizes differ.")
    if student_user_hidden.shape[1] != student_item_hs.shape[1]:
        raise ValueError("Student user/item hidden dimensions differ.")
    if teacher_user_hidden.shape[1] != teacher_item_hs.shape[1]:
        raise ValueError("Teacher user/item hidden dimensions differ.")

    candidate_mask, target_item_ids = (
        _build_privileged_item_candidate_mask(
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            history_item_ids=history_item_ids,
            target_item_ids=target_item_ids,
            topk=topk,
        )
    )
    distill_mask = distill_mask.reshape(-1).to(
        device=student_user_hidden.device,
        dtype=torch.bool,
    )
    if distill_mask.shape[0] != batch_size:
        raise ValueError("Item distillation mask and score batch sizes differ.")
    gate_weights = None
    if distill_weights is not None:
        gate_weights = distill_weights.reshape(-1).detach().to(
            device=student_user_hidden.device,
            dtype=torch.float32,
        )
        if gate_weights.shape[0] != batch_size:
            raise ValueError(
                "Item distillation weights and score batch sizes differ."
            )
        if not torch.isfinite(gate_weights).all() or (
            (gate_weights < 0) | (gate_weights > 1)
        ).any():
            raise ValueError(
                "Item distillation weights must be finite and in [0, 1]."
            )
        distill_mask = gate_weights > 0

    candidate_mask = candidate_mask.to(student_user_hidden.device)
    target_item_ids = target_item_ids.to(student_user_hidden.device)
    target_index = target_item_ids.unsqueeze(1)
    negative_mask = candidate_mask.clone()
    negative_mask.scatter_(1, target_index, False)

    # Restrict geometry to unique items in this mini-batch's candidate union.
    # For normal Top-K runs this avoids normalizing/copying the full item bank;
    # topk=0 naturally falls back to the complete available catalog.
    unique_item_ids = torch.unique(
        candidate_mask.nonzero(as_tuple=False)[:, 1],
        sorted=True,
    )
    target_columns = torch.searchsorted(unique_item_ids, target_item_ids)
    pair_rows, pair_item_ids = negative_mask.nonzero(as_tuple=True)
    pair_columns = torch.searchsorted(unique_item_ids, pair_item_ids)

    student_users = F.normalize(
        student_user_hidden.float(), p=2, dim=-1, eps=1e-12
    )
    student_items = F.normalize(
        student_item_hs.detach().index_select(
            0,
            unique_item_ids.to(student_item_hs.device),
        ).to(
            device=student_user_hidden.device,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    student_cosine = (student_users @ student_items.t()).clamp(-1.0, 1.0)
    student_distances = torch.sqrt(
        (2.0 - 2.0 * student_cosine).clamp_min(1e-12)
    )
    student_target_distance = student_distances.gather(
        1, target_columns.unsqueeze(1)
    ).squeeze(1)
    student_margins = (
        student_distances[pair_rows, pair_columns]
        - student_target_distance[pair_rows]
    )

    with torch.no_grad():
        teacher_users = F.normalize(
            teacher_user_hidden.detach().to(
                device=student_user_hidden.device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        teacher_items = F.normalize(
            teacher_item_hs.detach().index_select(
                0,
                unique_item_ids.to(teacher_item_hs.device),
            ).to(
                device=student_user_hidden.device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        teacher_cosine = (teacher_users @ teacher_items.t()).clamp(-1.0, 1.0)
        teacher_distances = torch.sqrt(
            (2.0 - 2.0 * teacher_cosine).clamp_min(1e-12)
        )
        teacher_target_distance = teacher_distances.gather(
            1, target_columns.unsqueeze(1)
        ).squeeze(1)
        teacher_margins = (
            teacher_distances[pair_rows, pair_columns]
            - teacher_target_distance[pair_rows]
        )

    eligible_pair_mask = distill_mask[pair_rows]
    pair_mask = eligible_pair_mask
    if positive_margin_only:
        pair_mask = pair_mask & (teacher_margins > 0)

    squared_error = (student_margins - teacher_margins).square()
    pair_counts = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=student_user_hidden.device,
    ).scatter_add(
        0,
        pair_rows,
        pair_mask.to(dtype=torch.long),
    )
    per_example_loss = torch.zeros(
        batch_size,
        dtype=squared_error.dtype,
        device=student_user_hidden.device,
    ).scatter_add(
        0,
        pair_rows,
        torch.where(
            pair_mask,
            squared_error,
            torch.zeros_like(squared_error),
        ),
    ) / pair_counts.clamp_min(1).to(squared_error.dtype)
    retained_examples = pair_counts > 0
    if gate_weights is not None:
        normalized_weights = gate_weights * retained_examples.to(
            dtype=gate_weights.dtype
        )
        weight_sum = normalized_weights.sum()
        safe_weight_sum = torch.where(
            weight_sum > 0,
            weight_sum,
            torch.ones_like(weight_sum),
        )
        loss = (
            per_example_loss * normalized_weights
        ).sum() / safe_weight_sum
    elif retained_examples.any():
        loss = per_example_loss[retained_examples].mean()
    else:
        loss = student_user_hidden.sum() * 0.0

    metric_zero = student_distances.detach()[:0].sum()
    if pair_mask.any():
        student_margin_mean = student_margins.detach()[pair_mask].mean()
        teacher_margin_mean = teacher_margins.detach()[pair_mask].mean()
    else:
        student_margin_mean = metric_zero
        teacher_margin_mean = metric_zero
    if retained_examples.any():
        geometry_rmse = per_example_loss.detach()[
            retained_examples
        ].mean().clamp_min(0.0).sqrt()
    else:
        geometry_rmse = metric_zero
    eligible_pair_count = eligible_pair_mask.sum().to(torch.float32)
    selected_pair_count = pair_mask.sum().to(torch.float32)
    metrics = {
        "opsd_item_kd_distilled_rate": distill_mask.float().mean(),
        "opsd_item_kd_candidate_count": (
            candidate_mask.float().sum(dim=-1).mean()
        ),
        "opsd_geometry_retained_example_rate": (
            retained_examples.float().mean()
        ),
        "opsd_geometry_pair_count": pair_counts.float().mean(),
        "opsd_geometry_teacher_positive_pair_rate": (
            selected_pair_count / eligible_pair_count.clamp_min(1.0)
        ),
        "opsd_geometry_student_margin_mean": student_margin_mean,
        "opsd_geometry_teacher_margin_mean": teacher_margin_mean,
        "opsd_geometry_margin_rmse": geometry_rmse,
    }
    return loss, metrics


def _privileged_item_rank_distillation_loss(
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        history_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        distill_mask: torch.Tensor,
        distill_weights=None,
        topk: int = 64,
        temperature: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Distil privileged-LLM ranking on a shared catalog candidate set.

    The detached teacher and trainable student are normalized over the union
    of teacher top-k, student top-k, and the current target item.  Historical
    items and padding id 0 are excluded.  ``distill_mask`` remains the
    OutcomeGRPO-reward gate, but accepted rows now receive the teacher's complete
    relative score distribution instead of a unit positive advantage.
    """

    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError(
            "Student/privileged-teacher scores must share shape [batch, items]."
        )
    if history_item_ids.ndim != 2:
        raise ValueError("History item ids must have shape [batch, history].")
    batch_size, num_items = student_scores.shape
    if history_item_ids.shape[0] != batch_size:
        raise ValueError("History and item-score batch sizes differ.")
    if num_items <= 1:
        raise ValueError("Item-ranking distillation needs at least two items.")
    if topk < 0:
        raise ValueError("opsd_rank_topk must be non-negative; use 0 for full catalog.")
    if temperature <= 0:
        raise ValueError("opsd_rank_temperature must be positive.")

    target_item_ids = target_item_ids.reshape(-1).to(student_scores.device)
    distill_mask = distill_mask.reshape(-1).to(
        device=student_scores.device,
        dtype=torch.bool,
    )
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("Target item ids and item-score batch sizes differ.")
    if distill_mask.shape[0] != batch_size:
        raise ValueError("Item distillation mask and score batch sizes differ.")
    gate_weights = None
    if distill_weights is not None:
        gate_weights = distill_weights.reshape(-1).detach().to(
            device=student_scores.device,
            dtype=torch.float32,
        )
        if gate_weights.shape[0] != batch_size:
            raise ValueError(
                "Item distillation weights and score batch sizes differ."
            )
        if not torch.isfinite(gate_weights).all() or (
            (gate_weights < 0) | (gate_weights > 1)
        ).any():
            raise ValueError(
                "Item distillation weights must be finite and in [0, 1]."
            )
        distill_mask = gate_weights > 0
    if ((target_item_ids <= 0) | (target_item_ids >= num_items)).any():
        raise ValueError("A target item id is outside the real item catalog.")

    history_item_ids = history_item_ids.to(student_scores.device)
    available = torch.ones(
        (batch_size, num_items),
        dtype=torch.bool,
        device=student_scores.device,
    )
    available[:, 0] = False
    valid_history = (
        (history_item_ids > 0) & (history_item_ids < num_items)
    )
    if valid_history.any():
        rows = torch.arange(
            batch_size, device=student_scores.device
        ).unsqueeze(1).expand_as(history_item_ids)
        available[rows[valid_history], history_item_ids[valid_history]] = False
    target_index = target_item_ids.unsqueeze(1)
    if (~available.gather(1, target_index)).any():
        raise ValueError("A target item is padding or already present in history.")
    if (~available).all(dim=-1).any():
        raise ValueError("An OPSD example has no recommendation candidates.")

    student_float = student_scores.float()
    teacher_float = teacher_scores.detach().to(student_scores.device).float()
    student_for_selection = student_float.detach().masked_fill(
        ~available, -torch.inf
    )
    teacher_for_selection = teacher_float.masked_fill(
        ~available, -torch.inf
    )

    if topk == 0:
        candidate_mask = available
    else:
        k = min(topk, num_items - 1)
        teacher_topk = torch.topk(
            teacher_for_selection, k=k, dim=-1
        ).indices
        student_topk = torch.topk(
            student_for_selection, k=k, dim=-1
        ).indices
        candidate_mask = torch.zeros_like(available)
        candidate_mask.scatter_(1, teacher_topk, True)
        candidate_mask.scatter_(1, student_topk, True)
        candidate_mask.scatter_(1, target_index, True)
        candidate_mask &= available

    fill_value = -1e4
    student_logits = (student_float / temperature).masked_fill(
        ~candidate_mask, fill_value
    )
    teacher_logits = (teacher_float / temperature).masked_fill(
        ~candidate_mask, fill_value
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    per_example_kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    if gate_weights is not None:
        weight_sum = gate_weights.sum()
        safe_weight_sum = torch.where(
            weight_sum > 0,
            weight_sum,
            torch.ones_like(weight_sum),
        )
        loss = (per_example_kl * gate_weights).sum() / safe_weight_sum
        if distill_mask.any():
            student_entropy = -(
                student_log_probs.exp() * student_log_probs
            ).sum(dim=-1)[distill_mask].mean()
            teacher_entropy = -(
                teacher_log_probs.exp() * teacher_log_probs
            ).sum(dim=-1)[distill_mask].mean()
        else:
            student_entropy = student_scores.detach().sum() * 0.0
            teacher_entropy = teacher_scores.detach().sum() * 0.0
    elif distill_mask.any():
        loss = per_example_kl[distill_mask].mean()
        student_entropy = -(
            student_log_probs.exp() * student_log_probs
        ).sum(dim=-1)[distill_mask].mean()
        teacher_entropy = -(
            teacher_log_probs.exp() * teacher_log_probs
        ).sum(dim=-1)[distill_mask].mean()
    else:
        loss = student_scores.sum() * 0.0
        student_entropy = student_scores.detach().sum() * 0.0
        teacher_entropy = teacher_scores.detach().sum() * 0.0

    metrics = {
        "opsd_item_kd_distilled_rate": distill_mask.float().mean(),
        "opsd_item_kd_candidate_count": candidate_mask.float().sum(dim=-1).mean(),
        "opsd_item_kd_student_entropy": student_entropy.detach(),
        "opsd_item_kd_teacher_entropy": teacher_entropy.detach(),
    }
    return loss, metrics


def _select_completion_prediction_hidden(
        sequence_hidden: torch.Tensor,
        labels: torch.Tensor,
):
    """Select hidden states that predict the labelled completion tokens."""

    if sequence_hidden.ndim != 3:
        raise ValueError("Sequence hidden states must have shape (batch, sequence, hidden).")
    if labels.ndim != 2:
        raise ValueError("OPSD labels must have shape (batch, sequence).")
    if sequence_hidden.shape[:2] != labels.shape:
        raise ValueError("Sequence hidden states and labels have different shapes.")

    completion_mask = labels[:, 1:] != -100
    prediction_hidden = sequence_hidden[:, :-1, :][completion_mask]
    target_ids = labels[:, 1:][completion_mask]
    per_example_counts = completion_mask.sum(dim=-1)
    return prediction_hidden, target_ids, per_example_counts


def _mean_reasoning_token_hidden(
        sequence_hidden: torch.Tensor,
        labels: torch.Tensor,
        marker_token_count: int,
):
    """Mean-pool labelled rollout tokens before the terminal marker.

    Unlike token KL, which consumes the hidden state immediately before each
    labelled token, rationale geometry represents the generated tokens
    themselves and therefore pools ``sequence_hidden`` at the labelled token
    positions. The final encoded ``emb_token`` positions are excluded so the
    recommendation readout marker cannot leak into the rationale summary.
    """

    if sequence_hidden.ndim != 3:
        raise ValueError(
            "Sequence hidden states must have shape (batch, sequence, hidden)."
        )
    if labels.ndim != 2 or sequence_hidden.shape[:2] != labels.shape:
        raise ValueError(
            "Sequence hidden states and rationale labels have different shapes."
        )
    marker_token_count = int(marker_token_count)
    if marker_token_count <= 0:
        raise ValueError("marker_token_count must be positive.")

    reasoning_mask = labels.to(sequence_hidden.device) != -100
    for row in range(reasoning_mask.shape[0]):
        completion_positions = torch.nonzero(
            reasoning_mask[row],
            as_tuple=False,
        ).flatten()
        if completion_positions.numel() <= marker_token_count:
            reasoning_mask[row].fill_(False)
            continue
        marker_positions = completion_positions[-marker_token_count:]
        reasoning_mask[row, marker_positions] = False

    reasoning_counts = reasoning_mask.sum(dim=-1)
    pooled_hidden = (
        sequence_hidden
        * reasoning_mask.unsqueeze(-1).to(sequence_hidden.dtype)
    ).sum(dim=1)
    pooled_hidden = pooled_hidden / reasoning_counts.clamp_min(1).to(
        sequence_hidden.dtype
    ).unsqueeze(1)
    valid_examples = reasoning_counts > 0
    return pooled_hidden, valid_examples, reasoning_counts


def _rationale_item_geometry_loss(
        rationale_hidden: torch.Tensor,
        target_item_hidden: torch.Tensor,
        target_item_ids: torch.Tensor,
        reason_temperature: float,
        item_temperature: float,
        loss_type: str = "relational_kl",
):
    """Match rationale relations to detached target-item relations.

    Each row defines a soft neighbourhood over the other examples in the
    current mini-batch. Target-item similarities supply the detached target
    geometry and rationale similarities supply the trainable geometry. The
    diagonal is excluded. ``relational_kl`` matches neighbourhood
    distributions; ``correlation`` minimizes one minus the Pearson correlation
    between the two off-diagonal similarity vectors.
    """

    if rationale_hidden.ndim != 2 or target_item_hidden.ndim != 2:
        raise ValueError(
            "Rationale and target-item hidden states must both be matrices."
        )
    batch_size = rationale_hidden.shape[0]
    target_item_ids = target_item_ids.reshape(-1)
    if (
        target_item_hidden.shape[0] != batch_size
        or target_item_ids.shape[0] != batch_size
    ):
        raise ValueError(
            "Rationale, target-item, and target-ID batch sizes differ."
        )
    loss_type = _normalize_opsd_rationale_geometry_loss_type(loss_type)
    reason_temperature = float(reason_temperature)
    item_temperature = float(item_temperature)
    if not math.isfinite(reason_temperature) or reason_temperature <= 0:
        raise ValueError(
            "opsd_rationale_reason_temperature must be finite and positive."
        )
    if not math.isfinite(item_temperature) or item_temperature <= 0:
        raise ValueError(
            "opsd_rationale_item_temperature must be finite and positive."
        )

    metric_zero = rationale_hidden.detach().reshape(-1)[:0].sum().float()
    if batch_size < 2:
        zero_loss = rationale_hidden.sum() * 0.0
        return zero_loss, {
            "opsd_rationale_geometry_pair_count": metric_zero,
            "opsd_rationale_geometry_duplicate_pair_rate": metric_zero,
            "opsd_rationale_geometry_item_similarity_mean": metric_zero,
            "opsd_rationale_geometry_reason_similarity_mean": metric_zero,
            "opsd_rationale_geometry_relation_correlation": metric_zero,
            "opsd_rationale_geometry_item_entropy": metric_zero,
            "opsd_rationale_geometry_reason_entropy": metric_zero,
        }

    rationale_vectors = F.normalize(
        rationale_hidden.float(),
        dim=-1,
        eps=1e-12,
    )
    item_vectors = F.normalize(
        target_item_hidden.detach().to(
            device=rationale_hidden.device,
            dtype=torch.float32,
        ),
        dim=-1,
        eps=1e-12,
    )
    reason_similarity = rationale_vectors @ rationale_vectors.transpose(0, 1)
    item_similarity = (
        item_vectors @ item_vectors.transpose(0, 1)
    ).detach()
    diagonal = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=rationale_hidden.device,
    )
    pair_mask = ~diagonal
    fill_value = -1e4
    reason_logits = (reason_similarity / reason_temperature).masked_fill(
        diagonal,
        fill_value,
    )
    item_logits = (item_similarity / item_temperature).masked_fill(
        diagonal,
        fill_value,
    )
    reason_log_probabilities = F.log_softmax(reason_logits, dim=-1)
    item_probabilities = F.softmax(item_logits, dim=-1).detach()
    reason_pair_values = reason_similarity[pair_mask]
    item_pair_values = item_similarity[pair_mask]
    reason_centered = reason_pair_values - reason_pair_values.mean()
    item_centered = item_pair_values - item_pair_values.mean()
    correlation_denominator = torch.sqrt(
        item_centered.square().sum()
        * reason_centered.square().sum()
    )
    relation_correlation = torch.where(
        correlation_denominator > 1e-12,
        (item_centered * reason_centered).sum()
        / correlation_denominator.clamp_min(1e-12),
        reason_pair_values.sum() * 0.0,
    )
    if loss_type == "correlation":
        # Undefined correlation (for example a two-sample batch or constant
        # target relations) carries no reliable geometry signal. Returning a
        # differentiable zero avoids turning it into an arbitrary unit loss.
        loss = torch.where(
            correlation_denominator > 1e-12,
            1.0 - relation_correlation,
            rationale_hidden.sum() * 0.0,
        )
    else:
        loss = F.kl_div(
            reason_log_probabilities,
            item_probabilities,
            reduction="batchmean",
        )

    with torch.no_grad():
        reason_probabilities = reason_log_probabilities.exp()
        detached_item_pair_values = item_pair_values.detach()
        detached_reason_pair_values = reason_pair_values.detach()
        ids = target_item_ids.to(rationale_hidden.device)
        duplicate_pairs = ids.unsqueeze(0) == ids.unsqueeze(1)
        duplicate_pair_rate = duplicate_pairs[pair_mask].float().mean()
        item_entropy = -(
            item_probabilities
            * item_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        reason_entropy = -(
            reason_probabilities
            * reason_log_probabilities
        ).sum(dim=-1).mean()
        metrics = {
            "opsd_rationale_geometry_pair_count": pair_mask.sum().float(),
            "opsd_rationale_geometry_duplicate_pair_rate": duplicate_pair_rate,
            "opsd_rationale_geometry_item_similarity_mean": (
                detached_item_pair_values.mean()
            ),
            "opsd_rationale_geometry_reason_similarity_mean": (
                detached_reason_pair_values.mean()
            ),
            "opsd_rationale_geometry_relation_correlation": (
                relation_correlation.detach()
            ),
            "opsd_rationale_geometry_item_entropy": item_entropy,
            "opsd_rationale_geometry_reason_entropy": reason_entropy,
        }
    return loss, {
        key: value.detach() for key, value in metrics.items()
    }


def _target_item_token_credit_weights(
        token_hidden: torch.Tensor,
        target_item_hidden: torch.Tensor,
        per_example_counts: torch.Tensor,
        temperature: float,
        mix: float,
):
    """Return detached, mean-one token weights from target-item relevance.

    Each example receives the same total weight as its completion length.  This
    preserves the scale of the original token mean while redistributing credit
    inside that example.  Credit is deliberately detached: it selects where
    the existing OPSD KL applies, rather than introducing a second trainable
    objective that can minimize the loss by collapsing its own weights.
    """

    if token_hidden.ndim != 2 or target_item_hidden.ndim != 2:
        raise ValueError(
            "Token and target-item hidden states must both be rank 2."
        )
    if token_hidden.shape[1] != target_item_hidden.shape[1]:
        raise ValueError(
            "Token and target-item hidden states have different dimensions."
        )
    if per_example_counts.ndim != 1:
        raise ValueError("Per-example token counts must be rank 1.")
    if target_item_hidden.shape[0] != per_example_counts.shape[0]:
        raise ValueError(
            "Target-item batch size does not match per-example token counts."
        )
    temperature = float(temperature)
    mix = float(mix)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "opsd_token_credit_temperature must be finite and positive."
        )
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("opsd_token_credit_mix must be in [0, 1].")

    counts = per_example_counts.to(
        device=token_hidden.device,
        dtype=torch.long,
    )
    if (counts < 0).any():
        raise ValueError("Per-example token counts must be non-negative.")
    if int(counts.sum().item()) != token_hidden.shape[0]:
        raise ValueError(
            "Per-example token counts do not match token hidden states."
        )

    with torch.no_grad():
        token_vectors = F.normalize(
            token_hidden.detach().float(), dim=-1, eps=1e-12
        )
        item_vectors = F.normalize(
            target_item_hidden.detach().to(
                device=token_hidden.device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=1e-12,
        )
        weights = token_vectors.new_empty(token_hidden.shape[0])
        entropies = []
        effective_counts = []
        effective_fractions = []
        top10_masses = []
        similarities = []
        offset = 0
        for example_index, count_tensor in enumerate(counts):
            count = int(count_tensor.item())
            if count == 0:
                continue
            end = offset + count
            scores = token_vectors[offset:end] @ item_vectors[example_index]
            attention = torch.softmax(scores / temperature, dim=0)
            weights[offset:end] = (
                (1.0 - mix) + mix * count * attention
            )

            if count > 1:
                entropy = -(
                    attention * attention.clamp_min(1e-12).log()
                ).sum() / math.log(count)
            else:
                entropy = attention.new_zeros(())
            effective_count = attention.square().sum().reciprocal()
            top_count = max(1, math.ceil(0.1 * count))
            entropies.append(entropy)
            effective_counts.append(effective_count)
            effective_fractions.append(effective_count / count)
            top10_masses.append(attention.topk(top_count).values.sum())
            similarities.append(scores)
            offset = end

        metric_zero = token_vectors[:0].sum()
        if weights.numel() > 0:
            weight_mean = weights.mean()
            weight_max = weights.max()
            similarity_values = torch.cat(similarities)
            similarity_mean = similarity_values.mean()
            similarity_std = similarity_values.std(unbiased=False)
        else:
            weight_mean = metric_zero
            weight_max = metric_zero
            similarity_mean = metric_zero
            similarity_std = metric_zero

        def _mean_or_zero(values):
            if values:
                return torch.stack(values).mean()
            return metric_zero

        metrics = {
            "opsd_token_credit_weight_mean": weight_mean,
            "opsd_token_credit_weight_max": weight_max,
            "opsd_token_credit_entropy": _mean_or_zero(entropies),
            "opsd_token_credit_effective_count": _mean_or_zero(
                effective_counts
            ),
            "opsd_token_credit_effective_fraction": _mean_or_zero(
                effective_fractions
            ),
            "opsd_token_credit_top10_mass": _mean_or_zero(top10_masses),
            "opsd_token_credit_similarity_mean": similarity_mean,
            "opsd_token_credit_similarity_std": similarity_std,
        }
    return weights.detach(), metrics


def _teacher_attention_token_credit_weights(
        last_layer_attention: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_token_count: int,
        mix: float,
):
    """Build detached mean-one credits from teacher readout attention.

    The returned vector follows the same flattened row-major completion order
    as ``_select_completion_prediction_hidden``.  Attention-derived mass is
    normalized only over reasoning tokens.  Terminal ``<answer>`` marker
    tokens retain only the uniform component, while the total weight of every
    example remains equal to its original completion length.
    """

    if last_layer_attention.ndim not in {3, 4}:
        raise ValueError(
            "Final-layer attention must have shape (batch, heads, key) or "
            "(batch, heads, query, key)."
        )
    if labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("OPSD labels/attention mask must both be rank 2.")
    if labels.shape != attention_mask.shape:
        raise ValueError("OPSD labels and attention mask have different shapes.")
    batch_size = last_layer_attention.shape[0]
    num_heads = last_layer_attention.shape[1]
    key_length = last_layer_attention.shape[-1]
    if batch_size != labels.shape[0]:
        raise ValueError("Teacher attention and label batch sizes differ.")
    if key_length != labels.shape[1]:
        raise ValueError(
            "Teacher attention sequence dimensions do not match labels."
        )
    if (
        last_layer_attention.ndim == 4
        and last_layer_attention.shape[-2] != labels.shape[1]
    ):
        raise ValueError(
            "Teacher attention query dimension does not match labels."
        )
    if num_heads <= 0:
        raise ValueError("Teacher attention must contain at least one head.")
    marker_token_count = int(marker_token_count)
    if marker_token_count <= 0:
        raise ValueError("marker_token_count must be positive.")
    mix = float(mix)
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("opsd_token_credit_mix must be in [0, 1].")

    with torch.no_grad():
        detached_attention = last_layer_attention.detach().float()
        if not torch.isfinite(detached_attention).all():
            raise ValueError("Teacher attention contains non-finite values.")

        valid_mask = attention_mask.to(
            device=detached_attention.device,
            dtype=torch.bool,
        )
        positions = torch.arange(
            key_length,
            device=detached_attention.device,
        ).unsqueeze(0).expand(batch_size, -1)
        readout_positions = positions.masked_fill(~valid_mask, -1).max(dim=1).values
        if (readout_positions < 0).any():
            raise ValueError("Teacher attention received an empty sequence.")
        batch_indices = torch.arange(
            batch_size,
            device=detached_attention.device,
        )
        if detached_attention.ndim == 4:
            readout_attention = detached_attention[
                batch_indices,
                :,
                readout_positions,
                :,
            ].mean(dim=1)
        else:
            readout_attention = detached_attention.mean(dim=1)

        completion_mask = labels.to(detached_attention.device) != -100
        completion_mask &= valid_mask
        # Causal logits at position j-1 predict labels[:, j], so position zero
        # can never appear in the flattened per-token KL vector.
        completion_mask[:, 0] = False

        all_weights = []
        entropies = []
        effective_counts = []
        effective_fractions = []
        top10_masses = []
        reasoning_masses = []
        metric_zero = detached_attention[:0].sum()
        for example_index in range(batch_size):
            completion_positions = torch.nonzero(
                completion_mask[example_index],
                as_tuple=False,
            ).flatten()
            completion_count = int(completion_positions.numel())
            if completion_count == 0:
                continue

            token_weights = readout_attention.new_full(
                (completion_count,),
                1.0 - mix,
            )
            reasoning_count = completion_count - marker_token_count
            if reasoning_count <= 0:
                # Malformed/degenerate completions should not silently erase
                # the token KL. Fall back to uniform credit for this example.
                token_weights.fill_(1.0)
                normalized_attention = None
            else:
                reasoning_positions = completion_positions[:reasoning_count]
                reasoning_scores = readout_attention[
                    example_index,
                    reasoning_positions,
                ].clamp_min(0.0)
                reasoning_mass = reasoning_scores.sum()
                if not torch.isfinite(reasoning_mass) or reasoning_mass <= 0:
                    normalized_attention = torch.full_like(
                        reasoning_scores,
                        1.0 / reasoning_count,
                    )
                else:
                    normalized_attention = reasoning_scores / reasoning_mass
                token_weights[:reasoning_count] += (
                    mix * completion_count * normalized_attention
                )
                reasoning_masses.append(reasoning_mass)

            all_weights.append(token_weights)
            if normalized_attention is None:
                continue
            if reasoning_count > 1:
                entropy = -(
                    normalized_attention
                    * normalized_attention.clamp_min(1e-12).log()
                ).sum() / math.log(reasoning_count)
            else:
                entropy = normalized_attention.new_zeros(())
            effective_count = normalized_attention.square().sum().reciprocal()
            top_count = max(1, math.ceil(0.1 * reasoning_count))
            entropies.append(entropy)
            effective_counts.append(effective_count)
            effective_fractions.append(effective_count / reasoning_count)
            top10_masses.append(
                normalized_attention.topk(top_count).values.sum()
            )

        if all_weights:
            weights = torch.cat(all_weights)
            weight_mean = weights.mean()
            weight_max = weights.max()
        else:
            weights = readout_attention.new_empty(0)
            weight_mean = metric_zero
            weight_max = metric_zero

        def _mean_or_zero(values):
            if values:
                return torch.stack(values).mean()
            return metric_zero

        metrics = {
            "opsd_token_credit_weight_mean": weight_mean,
            "opsd_token_credit_weight_max": weight_max,
            "opsd_token_credit_entropy": _mean_or_zero(entropies),
            "opsd_token_credit_effective_count": _mean_or_zero(
                effective_counts
            ),
            "opsd_token_credit_effective_fraction": _mean_or_zero(
                effective_fractions
            ),
            "opsd_token_credit_top10_mass": _mean_or_zero(top10_masses),
            "opsd_token_credit_attention_reasoning_mass": _mean_or_zero(
                reasoning_masses
            ),
        }
    return weights.detach(), {
        key: value.detach() for key, value in metrics.items()
    }


def _last_non_padding_hidden(
        sequence_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the final attended token without assuming a padding side."""

    if sequence_hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Hidden states/attention mask must be rank 3/rank 2.")
    if sequence_hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and attention mask have different shapes.")

    valid_mask = attention_mask.to(dtype=torch.bool)
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0).expand_as(attention_mask)
    last_positions = positions.masked_fill(~valid_mask, -1).amax(dim=-1)
    if (last_positions < 0).any():
        raise ValueError("An OPSD sequence contains no attended token.")
    rows = torch.arange(sequence_hidden.shape[0], device=sequence_hidden.device)
    return sequence_hidden[rows, last_positions]


def _token_divergence_from_logits(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
        loss_type: str,
) -> torch.Tensor:
    """Return one detached-teacher divergence value per token position."""

    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("Selected OPSD logits must have shape (tokens, vocabulary).")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Selected student and teacher logits must have the same shape.")
    if temperature <= 0:
        raise ValueError("opsd_temperature must be positive.")

    divergence = loss_type.lower()
    if divergence not in {"jsd", "kl", "reverse_kl"}:
        raise ValueError(
            f"Unsupported opsd_loss_type={loss_type!r}; use 'kl', "
            "'reverse_kl', or 'jsd'."
        )

    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )

    if divergence == "kl":
        per_token_loss = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    elif divergence == "reverse_kl":
        student_probs = student_log_probs.exp()
        per_token_loss = (
            student_probs
            * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)
    else:
        midpoint_log_probs = torch.logaddexp(
            student_log_probs, teacher_log_probs
        ) - math.log(2.0)
        student_to_midpoint = F.kl_div(
            midpoint_log_probs,
            student_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        teacher_to_midpoint = F.kl_div(
            midpoint_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        per_token_loss = 0.5 * (student_to_midpoint + teacher_to_midpoint)

    return per_token_loss * (temperature ** 2)


def _entropy_gap_token_gate(
        student_entropy: torch.Tensor,
        teacher_entropy: torch.Tensor,
        teacher_topk_mass: torch.Tensor,
        gap_threshold: float,
        min_teacher_topk_mass: float,
):
    """Select tokens using a detached entropy gap and mass reliability gate.

    Entropies are conditional on the shared candidate support used by the
    token divergence. Positions whose original teacher Top-K mass is below the
    reliability threshold receive zero distillation weight.
    """

    if (
        student_entropy.ndim != 1
        or teacher_entropy.ndim != 1
        or teacher_topk_mass.ndim != 1
    ):
        raise ValueError("Entropy-gate inputs must be rank-1 tensors.")
    if not (
        student_entropy.shape
        == teacher_entropy.shape
        == teacher_topk_mass.shape
    ):
        raise ValueError("Entropy-gate inputs must have the same shape.")
    gap_threshold = float(gap_threshold)
    if not math.isfinite(gap_threshold) or gap_threshold < 0:
        raise ValueError(
            "Entropy gap threshold must be finite and non-negative."
        )
    min_teacher_topk_mass = float(min_teacher_topk_mass)
    if (
        not math.isfinite(min_teacher_topk_mass)
        or min_teacher_topk_mass < 0
        or min_teacher_topk_mass > 1
    ):
        raise ValueError("Teacher Top-K mass threshold must be in [0, 1].")

    student_values = student_entropy.detach().float()
    teacher_values = teacher_entropy.detach().to(
        device=student_values.device,
        dtype=torch.float32,
    )
    mass_values = teacher_topk_mass.detach().to(
        device=student_values.device,
        dtype=torch.float32,
    )
    if not (
        torch.isfinite(student_values).all()
        and torch.isfinite(teacher_values).all()
        and torch.isfinite(mass_values).all()
    ):
        raise ValueError("Entropy-gate inputs must be finite.")
    # The mass is computed as exp(logsumexp(top-k) - logsumexp(full vocab)).
    # It is mathematically in [0, 1]. Allow an upper safety band through 1.1
    # for mixed-precision reduction noise, then clamp before gate decisions.
    lower_mass_tolerance = 1e-5
    upper_mass_safety_bound = 1.1
    if (
        (mass_values < -lower_mass_tolerance)
        | (mass_values > upper_mass_safety_bound)
    ).any():
        raise ValueError(
            "Teacher Top-K masses must be non-negative and no greater than "
            f"the numerical safety bound {upper_mass_safety_bound}."
        )
    mass_values = mass_values.clamp(0.0, 1.0)

    entropy_gap = student_values - teacher_values
    reliable = mass_values >= min_teacher_topk_mass
    entropy_selected = entropy_gap > gap_threshold
    low_mass_filtered = ~reliable
    keep = entropy_selected & reliable

    metric_zero = entropy_gap[:0].sum()
    if entropy_gap.numel() > 0:
        candidate_token_count = entropy_gap.new_tensor(
            float(entropy_gap.numel())
        )
        active_token_count = keep.sum().to(dtype=torch.float32)
        filtered_token_count = (~keep).sum().to(dtype=torch.float32)
        metrics = {
            "opsd_student_token_entropy_mean": student_values.mean(),
            "opsd_teacher_token_entropy_mean": teacher_values.mean(),
            "opsd_entropy_gap_mean": entropy_gap.mean(),
            "opsd_entropy_gap_positive_rate": (entropy_gap > 0).float().mean(),
            "opsd_entropy_gate_threshold_pass_rate": (
                entropy_selected.float().mean()
            ),
            "opsd_entropy_gate_reliable_rate": reliable.float().mean(),
            "opsd_entropy_gate_low_mass_filtered_rate": (
                low_mass_filtered.float().mean()
            ),
            "opsd_entropy_gate_keep_rate": keep.float().mean(),
            "opsd_entropy_gate_candidate_token_count": candidate_token_count,
            "opsd_entropy_gate_active_token_count": active_token_count,
            "opsd_entropy_gate_filtered_token_count": filtered_token_count,
        }
    else:
        metrics = {
            "opsd_student_token_entropy_mean": metric_zero,
            "opsd_teacher_token_entropy_mean": metric_zero,
            "opsd_entropy_gap_mean": metric_zero,
            "opsd_entropy_gap_positive_rate": metric_zero,
            "opsd_entropy_gate_threshold_pass_rate": metric_zero,
            "opsd_entropy_gate_reliable_rate": metric_zero,
            "opsd_entropy_gate_low_mass_filtered_rate": metric_zero,
            "opsd_entropy_gate_keep_rate": metric_zero,
            "opsd_entropy_gate_candidate_token_count": metric_zero,
            "opsd_entropy_gate_active_token_count": metric_zero,
            "opsd_entropy_gate_filtered_token_count": metric_zero,
        }
    return keep.detach(), {
        key: value.detach() for key, value in metrics.items()
    }


def _entropy_gap_token_weights(
        student_entropy: torch.Tensor,
        teacher_entropy: torch.Tensor,
        teacher_topk_mass: torch.Tensor,
        gap_threshold: float,
        min_teacher_topk_mass: float,
        mode: str,
        temperature: float,
):
    """Return detached weights, zeroing unreliable low-mass positions."""

    mode = _normalize_opsd_entropy_gate_mode(mode)
    if mode == "none":
        raise ValueError(
            "Entropy token weights require hard or soft gate mode."
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Entropy gate temperature must be finite and positive."
        )

    hard_keep, metrics = _entropy_gap_token_gate(
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy,
        teacher_topk_mass=teacher_topk_mass,
        gap_threshold=gap_threshold,
        min_teacher_topk_mass=min_teacher_topk_mass,
    )
    if mode == "hard":
        weights = hard_keep.to(dtype=torch.float32)
    else:
        student_values = student_entropy.detach().float()
        teacher_values = teacher_entropy.detach().to(
            device=student_values.device,
            dtype=torch.float32,
        )
        mass_values = teacher_topk_mass.detach().to(
            device=student_values.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        positive_gap = (
            student_values - teacher_values - float(gap_threshold)
        ).clamp_min(0.0)
        soft_weights = 1.0 - torch.exp(-positive_gap / temperature)
        reliable = mass_values >= float(min_teacher_topk_mass)
        weights = torch.where(
            reliable,
            soft_weights,
            torch.zeros_like(soft_weights),
        )

    metric_zero = weights[:0].sum()
    if weights.numel() > 0:
        metrics.update({
            "opsd_entropy_gate_weight_mean": weights.mean(),
            "opsd_entropy_gate_weight_p50": torch.quantile(weights, 0.5),
            "opsd_entropy_gate_weight_p90": torch.quantile(weights, 0.9),
        })
    else:
        metrics.update({
            "opsd_entropy_gate_weight_mean": metric_zero,
            "opsd_entropy_gate_weight_p50": metric_zero,
            "opsd_entropy_gate_weight_p90": metric_zero,
        })
    return weights.detach(), {
        key: value.detach() for key, value in metrics.items()
    }


def _completion_token_distillation_loss(
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        student_output_head,
        teacher_output_head,
        temperature: float,
        loss_type: str,
        topk: int,
        chunk_size: int,
        return_per_token: bool = False,
        return_teacher_topk_mass: bool = False,
        return_token_entropies: bool = False,
) -> torch.Tensor:
    """Distil completion positions with a teacher-top-k student projection.

    The teacher must still evaluate its full vocabulary in each chunk to find
    the exact top-k.  The student is projected only onto those selected output
    rows.  The sampled target token is forced into the candidate set so the
    on-policy action always receives a training signal.
    """

    if student_hidden.ndim != 2 or teacher_hidden.ndim != 2:
        raise ValueError("Selected completion hidden states must be rank 2.")
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("Student and teacher completion hidden shapes differ.")
    if target_ids.ndim != 1 or target_ids.shape[0] != student_hidden.shape[0]:
        raise ValueError("Completion target ids do not match selected hidden states.")
    if chunk_size <= 0:
        raise ValueError("opsd_token_chunk_size must be positive.")
    if topk < 0:
        raise ValueError("opsd_topk must be non-negative.")
    if not hasattr(student_output_head, "weight"):
        raise TypeError("The student output head must expose a weight tensor.")
    if not hasattr(teacher_output_head, "weight"):
        raise TypeError("The teacher output head must expose a weight tensor.")

    if student_hidden.shape[0] == 0:
        empty = student_hidden.reshape(-1)[:0].float()
        if return_per_token:
            # Preserve a zero-gradient connection to the student so an
            # all-filtered mini-batch remains safe to backpropagate.
            loss_result = empty
        else:
            loss_result = student_hidden.sum() * 0.0
        if return_teacher_topk_mass and return_token_entropies:
            return (
                loss_result,
                empty.detach(),
                empty.detach(),
                empty.detach(),
            )
        if return_teacher_topk_mass:
            return loss_result, empty.detach()
        if return_token_entropies:
            return loss_result, empty.detach(), empty.detach()
        return loss_result

    vocab_size = student_output_head.weight.shape[0]
    teacher_vocab_size = teacher_output_head.weight.shape[0]
    if teacher_vocab_size != vocab_size:
        raise ValueError(
            "Student and teacher output heads must use the same vocabulary."
        )
    if topk > vocab_size:
        raise ValueError(
            f"opsd_topk={topk} exceeds model vocabulary size {vocab_size}."
        )
    if topk == 1:
        raise ValueError("opsd_topk=1 has a zero-valued normalized divergence; use >=2.")

    total = student_hidden.new_zeros((), dtype=torch.float32)
    total_count = 0
    per_token_values = [] if return_per_token else None
    teacher_topk_masses = [] if return_teacher_topk_mass else None
    student_entropies = [] if return_token_entropies else None
    teacher_entropies = [] if return_token_entropies else None
    for start in range(0, student_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, student_hidden.shape[0])
        student_chunk = student_hidden[start:end]
        teacher_chunk = teacher_hidden[start:end]
        # PEFT/model normalization can expose float32 hidden states even when
        # the frozen LM head remains bfloat16.  Linear/bmm require matching
        # dtypes; casting the small hidden chunks avoids materialising a
        # float32 copy of the full vocabulary head and preserves gradients
        # through the student cast during normal OPSD training.
        student_head_dtype = student_output_head.weight.dtype
        teacher_head_dtype = teacher_output_head.weight.dtype
        student_head_chunk = student_chunk.to(dtype=student_head_dtype)
        teacher_head_chunk = teacher_chunk.to(dtype=teacher_head_dtype)

        if topk == 0:
            student_logits = student_output_head(student_head_chunk)
            with torch.no_grad():
                teacher_logits = teacher_output_head(teacher_head_chunk)
                if teacher_topk_masses is not None:
                    teacher_topk_masses.append(
                        torch.ones(
                            teacher_logits.shape[0],
                            device=teacher_logits.device,
                            dtype=torch.float32,
                        )
                    )
        else:
            with torch.no_grad():
                teacher_full_logits = teacher_output_head(teacher_head_chunk)
                candidate_ids = torch.topk(
                    teacher_full_logits, k=topk, dim=-1
                ).indices
                if teacher_topk_masses is not None:
                    scaled_teacher_full_logits = (
                        teacher_full_logits.float() / float(temperature)
                    )
                    original_topk_logits = teacher_full_logits.gather(
                        1, candidate_ids
                    ).float() / float(temperature)
                    full_log_normalizer = torch.logsumexp(
                        scaled_teacher_full_logits, dim=-1
                    )
                    teacher_topk_masses.append(torch.exp(
                        torch.logsumexp(original_topk_logits, dim=-1)
                        - full_log_normalizer
                    ))
                chunk_targets = target_ids[start:end]
                target_is_present = (
                    candidate_ids == chunk_targets.unsqueeze(-1)
                ).any(dim=-1)
                if not target_is_present.all():
                    candidate_ids = candidate_ids.clone()
                    candidate_ids[~target_is_present, -1] = chunk_targets[
                        ~target_is_present
                    ]
                teacher_logits = teacher_full_logits.gather(1, candidate_ids)

            selected_weight = student_output_head.weight[candidate_ids]
            student_logits = torch.bmm(
                selected_weight,
                student_head_chunk.unsqueeze(-1),
            ).squeeze(-1)
            output_bias = getattr(student_output_head, "bias", None)
            if output_bias is not None:
                student_logits = student_logits + output_bias[candidate_ids]

        if student_entropies is not None:
            with torch.no_grad():
                student_log_probs = F.log_softmax(
                    student_logits.detach().float() / temperature,
                    dim=-1,
                )
                teacher_log_probs = F.log_softmax(
                    teacher_logits.detach().float() / temperature,
                    dim=-1,
                )
                student_entropies.append(-(
                    student_log_probs.exp() * student_log_probs
                ).sum(dim=-1))
                teacher_entropies.append(-(
                    teacher_log_probs.exp() * teacher_log_probs
                ).sum(dim=-1))

        per_token_loss = _token_divergence_from_logits(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=temperature,
            loss_type=loss_type,
        )
        total = total + per_token_loss.sum()
        total_count += per_token_loss.numel()

        if per_token_values is not None:
            per_token_values.append(per_token_loss)

    if per_token_values is not None:
        loss_result = torch.cat(per_token_values, dim=0)
    else:
        loss_result = total / total_count
    if teacher_topk_masses is not None and student_entropies is not None:
        return (
            loss_result,
            torch.cat(teacher_topk_masses, dim=0).detach(),
            torch.cat(student_entropies, dim=0).detach(),
            torch.cat(teacher_entropies, dim=0).detach(),
        )
    if teacher_topk_masses is not None:
        return loss_result, torch.cat(teacher_topk_masses, dim=0).detach()
    if student_entropies is not None:
        return (
            loss_result,
            torch.cat(student_entropies, dim=0).detach(),
            torch.cat(teacher_entropies, dim=0).detach(),
        )
    return loss_result


def _reasoning_position_mask_from_counts(
        per_example_counts: torch.Tensor,
        marker_token_count: int,
):
    """Mask flattened completion positions while excluding terminal markers."""

    if per_example_counts.ndim != 1:
        raise ValueError("Per-example token counts must be rank 1.")
    marker_token_count = int(marker_token_count)
    if marker_token_count <= 0:
        raise ValueError("marker_token_count must be positive.")
    counts = per_example_counts.to(dtype=torch.long)
    if (counts < 0).any():
        raise ValueError("Per-example token counts must be non-negative.")

    mask = torch.ones(
        int(counts.sum().item()),
        dtype=torch.bool,
        device=counts.device,
    )
    offset = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        end = offset + count
        marker_start = max(offset, end - marker_token_count)
        mask[marker_start:end] = False
        offset = end
    return mask


def _teacher_topk_mass_summary(masses: torch.Tensor):
    """Summarize detached teacher Top-K probability masses exactly."""

    if masses.ndim != 1:
        raise ValueError("Teacher Top-K probability masses must be rank 1.")
    values = masses.detach().float().cpu()
    if values.numel() == 0:
        return {
            "opsd_teacher_topk_mass_sample_count": 0.0,
            "opsd_teacher_topk_mass_mean": 0.0,
            "opsd_teacher_topk_mass_median": 0.0,
            "opsd_teacher_topk_mass_p05": 0.0,
            "opsd_teacher_topk_mass_p01": 0.0,
            "opsd_teacher_topk_mass_minimum": 0.0,
            **{
                f"opsd_teacher_topk_mass_lt_{int(threshold * 100):03d}_rate": 0.0
                for threshold in (0.75, 0.80, 0.90, 0.95, 0.99)
            },
        }
    if not torch.isfinite(values).all():
        raise ValueError("Teacher Top-K probability masses must be finite.")
    if (values < -1e-6).any() or (values > 1.0 + 1e-6).any():
        raise ValueError("Teacher Top-K probability masses must be in [0, 1].")
    values = values.clamp(0.0, 1.0)
    quantiles = torch.quantile(
        values,
        torch.tensor([0.50, 0.05, 0.01], dtype=torch.float32),
    )
    metrics = {
        "opsd_teacher_topk_mass_sample_count": float(values.numel()),
        "opsd_teacher_topk_mass_mean": float(values.mean().item()),
        "opsd_teacher_topk_mass_median": float(quantiles[0].item()),
        "opsd_teacher_topk_mass_p05": float(quantiles[1].item()),
        "opsd_teacher_topk_mass_p01": float(quantiles[2].item()),
        "opsd_teacher_topk_mass_minimum": float(values.min().item()),
    }
    for threshold in (0.75, 0.80, 0.90, 0.95, 0.99):
        metrics[
            f"opsd_teacher_topk_mass_lt_{int(threshold * 100):03d}_rate"
        ] = float((values < threshold).float().mean().item())
    return metrics


def _filter_token_kl_by_example(
        per_token_kl: torch.Tensor,
        per_example_counts: torch.Tensor,
        max_token_kl: float,
        return_selection: bool = False,
):
    """Drop whole examples whose detached mean token KL exceeds a limit."""

    if per_token_kl.ndim != 1:
        raise ValueError("Per-token OPSD KL must be a rank-1 tensor.")
    if per_example_counts.ndim != 1:
        raise ValueError("Per-example OPSD token counts must be rank 1.")
    if not math.isfinite(max_token_kl) or max_token_kl <= 0:
        raise ValueError("opsd_max_token_kl must be finite and positive.")

    counts = per_example_counts.to(
        device=per_token_kl.device,
        dtype=torch.long,
    )
    if (counts < 0).any():
        raise ValueError("Per-example OPSD token counts must be non-negative.")
    if int(counts.sum().item()) != per_token_kl.numel():
        raise ValueError(
            "Per-example OPSD token counts do not match per-token KL values."
        )

    num_examples = counts.shape[0]
    example_ids = torch.repeat_interleave(
        torch.arange(num_examples, device=per_token_kl.device),
        counts,
    )
    per_example_sums = per_token_kl.new_zeros(num_examples)
    if per_token_kl.numel() > 0:
        per_example_sums = per_example_sums.scatter_add(
            0,
            example_ids,
            per_token_kl,
        )

    eligible_examples = counts > 0
    safe_counts = counts.clamp_min(1).to(per_token_kl.dtype)
    per_example_mean = per_example_sums / safe_counts
    detached_mean = per_example_mean.detach()
    finite_examples = eligible_examples & torch.isfinite(detached_mean)
    kept_examples = finite_examples & (detached_mean <= max_token_kl)
    kept_tokens = kept_examples[example_ids]
    selected_token_kl = per_token_kl[kept_tokens]
    if selected_token_kl.numel() > 0:
        filtered_loss = selected_token_kl.mean()
    else:
        # Avoid `inf * 0 -> nan` while retaining a zero-gradient graph edge.
        filtered_loss = per_token_kl[:0].sum()

    metric_zero = per_token_kl.detach()[:0].sum()
    finite_values = detached_mean[finite_examples]
    if finite_values.numel() > 0:
        mean_before_filter = finite_values.mean()
        max_before_filter = finite_values.max()
    else:
        mean_before_filter = metric_zero
        max_before_filter = metric_zero

    eligible_count = eligible_examples.sum().to(dtype=torch.float32)
    kept_count = kept_examples.sum().to(dtype=torch.float32)
    denominator = eligible_count.clamp_min(1.0)
    keep_rate = kept_count / denominator
    metrics = {
        "opsd_token_kl_mean_before_filter": mean_before_filter,
        "opsd_token_kl_max_before_filter": max_before_filter,
        "opsd_token_kl_filter_keep_rate": keep_rate,
    }
    if return_selection:
        return filtered_loss, metrics, kept_tokens, kept_examples
    return filtered_loss, metrics


def _build_train_positive_item_map(records):
    """Collect unique positive item IDs per user from training rows only."""

    positive_ids_by_user = {}
    for record in records:
        item_id = int(record["seq_labels"])
        if item_id <= 0:
            continue
        user_key = str(record["user_id"])
        positive_ids_by_user.setdefault(user_key, set()).add(item_id)
    return {
        user_key: tuple(sorted(item_ids))
        for user_key, item_ids in positive_ids_by_user.items()
    }


def _format_hard_negative_privileged_context(
        target_title,
        hard_negative_titles,
):
    """Format compact target-vs-alternative context for the privileged teacher."""

    target_title = str(target_title).strip()
    alternatives = []
    seen = {target_title}
    for title in hard_negative_titles:
        title = str(title).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        alternatives.append(title)

    if alternatives:
        alternative_lines = "\n".join(
            f"[{title}]" for title in alternatives
        )
    else:
        alternative_lines = "(no safe alternative available)"

    return (
        "Training-only comparison information:\n\n"
        "Observed next item:\n"
        f"[{target_title}]\n\n"
        "Other items currently confusable for the student:\n"
        f"{alternative_lines}\n\n"
        "Use the user's history to focus on preference evidence that "
        "distinguishes the observed item from these alternatives. "
        "Keep the original response format."
    )


def _mine_student_hard_negative_ids(
        student_scores,
        history_item_ids,
        target_item_ids,
        known_positive_item_ids,
        topm,
):
    """Select detached student Top-M items after recommendation-safe masking."""

    if student_scores.ndim != 2:
        raise ValueError("student_scores must have shape [batch, items].")
    if history_item_ids.ndim != 2:
        raise ValueError("history item IDs must have shape [batch, history].")
    if topm <= 0:
        raise ValueError("opsd_hard_negative_count must be positive.")

    scores = student_scores.detach().float()
    batch_size, num_items = scores.shape
    if num_items <= 1:
        raise ValueError("Hard-negative mining requires real catalog items.")
    if history_item_ids.shape[0] != batch_size:
        raise ValueError("history and student score batch sizes differ.")

    target_item_ids = target_item_ids.reshape(-1).to(
        device=scores.device,
        dtype=torch.long,
    )
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("target and student score batch sizes differ.")
    if ((target_item_ids <= 0) | (target_item_ids >= num_items)).any():
        raise ValueError("A hard-negative target is outside the real catalog.")

    available = torch.ones(
        (batch_size, num_items),
        dtype=torch.bool,
        device=scores.device,
    )
    available[:, 0] = False

    def _mask_ids(item_ids, label):
        if item_ids is None:
            return scores.new_tensor(0.0)
        if item_ids.ndim != 2 or item_ids.shape[0] != batch_size:
            raise ValueError(
                f"{label} item IDs must have shape [batch, items]."
            )
        item_ids = item_ids.to(device=scores.device, dtype=torch.long)
        valid = (item_ids > 0) & (item_ids < num_items)
        if valid.any():
            rows = torch.arange(
                batch_size,
                device=scores.device,
            ).unsqueeze(1).expand_as(item_ids)
            available[rows[valid], item_ids[valid]] = False
        return valid.sum().to(dtype=torch.float32)

    _mask_ids(history_item_ids, "history")
    known_positive_count = _mask_ids(
        known_positive_item_ids,
        "known-positive",
    )
    available.scatter_(1, target_item_ids.unsqueeze(1), False)

    k = min(int(topm), num_items - 1)
    masked_scores = scores.masked_fill(~available, -torch.inf)
    top_values, hard_negative_ids = torch.topk(masked_scores, k=k, dim=-1)
    valid_selected = torch.isfinite(top_values)
    hard_negative_ids = hard_negative_ids.masked_fill(~valid_selected, 0)
    if k < topm:
        padding = torch.zeros(
            (batch_size, topm - k),
            dtype=hard_negative_ids.dtype,
            device=hard_negative_ids.device,
        )
        hard_negative_ids = torch.cat((hard_negative_ids, padding), dim=1)

    selected_counts = valid_selected.sum(dim=-1)
    has_alternative = selected_counts > 0
    metric_zero = scores[:, :0].sum()
    if has_alternative.any():
        target_scores = scores.gather(1, target_item_ids.unsqueeze(1)).squeeze(1)
        margins = target_scores[has_alternative] - top_values[has_alternative, 0]
        target_margin = margins.mean()
        target_above_rate = (margins > 0).float().mean()
    else:
        target_margin = metric_zero
        target_above_rate = metric_zero

    metrics = {
        "opsd_hn_selected_count": selected_counts.float().mean(),
        "opsd_hn_shortfall_rate": (selected_counts < topm).float().mean(),
        "opsd_hn_student_target_margin": target_margin,
        "opsd_hn_student_target_above_hardest_rate": target_above_rate,
        "opsd_hn_known_train_positive_ids": (
            known_positive_count / max(1, batch_size)
        ),
    }
    return hard_negative_ids.detach(), {
        key: value.detach() for key, value in metrics.items()
    }


def _build_train_future_item_map(records, max_items: int) -> Dict[int, tuple]:
    """Index bounded, chronological future items using training rows only."""

    if max_items <= 0:
        raise ValueError(
            "dataset_window_size must be positive when "
            "opsd_privileged_context=future."
        )

    events_by_user = {}
    seen_train_data_ids = set()
    for record in records:
        train_data_id = int(record["train_data_id"])
        if train_data_id in seen_train_data_ids:
            raise ValueError(
                f"Duplicate train_data_id in OPSD future index: {train_data_id}"
            )
        seen_train_data_ids.add(train_data_id)

        target_item_id = int(record["seq_labels"])
        timestamp = float(record["timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError(
                f"Non-finite timestamp for train_data_id={train_data_id}."
            )
        interaction_id = record.get("interaction_id", train_data_id)
        if interaction_id is None:
            interaction_id = train_data_id
        interaction_id = int(interaction_id)
        title = str(record.get("item_title") or f"Item {target_item_id}")
        rating = record.get("rating")

        user_key = str(record["user_id"])
        events_by_user.setdefault(user_key, []).append(
            (
                timestamp,
                interaction_id,
                train_data_id,
                target_item_id,
                title,
                rating,
            )
        )

    future_items_by_train_data_id = {}
    for events in events_by_user.values():
        events.sort(key=lambda event: event[:3])
        for position, event in enumerate(events):
            current_train_data_id = event[2]
            current_target_item_id = event[3]
            future_items = []
            for later_event in events[position + 1:]:
                later_item_id = later_event[3]
                if later_item_id <= 0 or later_item_id == current_target_item_id:
                    continue
                future_items.append(
                    (later_item_id, later_event[4], later_event[5])
                )
                if len(future_items) == max_items:
                    break
            if future_items:
                future_items_by_train_data_id[current_train_data_id] = tuple(
                    future_items
                )

    return future_items_by_train_data_id


class OPSDOutcomeGRPOTrainer(OutcomeGRPOTrainer):
    """OutcomeGRPO with selectable target recommendation and online distillation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opsd_item_bank_last_refresh_step = None
        self._opsd_frozen_teacher_model = None
        self._opsd_frozen_teacher_item_hs = None
        self._opsd_teacher_last_sync_step = 0
        self._opsd_teacher_sync_count = 0
        self._opsd_teacher_last_update_norm = 0.0
        self._opsd_lagged_parameter_names = ()
        self._opsd_topk_mass_sample_chunks = []
        self._opsd_topk_mass_observed_positions = 0
        self._opsd_topk_mass_retained_positions = 0

        topk_mass_diagnostics = bool(getattr(
            self.args,
            "opsd_topk_mass_diagnostics",
            False,
        ))
        topk_mass_max_positions = int(getattr(
            self.args,
            "opsd_topk_mass_max_positions",
            100000,
        ))
        if topk_mass_max_positions <= 0:
            raise ValueError(
                "opsd_topk_mass_max_positions must be positive."
            )
        if (
            topk_mass_diagnostics
            and float(self.args.opsd_token_weight) == 0
        ):
            raise ValueError(
                "Teacher Top-K mass diagnostics require a non-zero "
                "opsd_token_weight."
            )
        self.args.opsd_topk_mass_diagnostics = topk_mass_diagnostics
        self.args.opsd_topk_mass_max_positions = topk_mass_max_positions

        entropy_gate_mode = _normalize_opsd_entropy_gate_mode(getattr(
            self.args,
            "opsd_entropy_gate_mode",
            "none",
        ))
        entropy_gap_threshold = float(getattr(
            self.args,
            "opsd_entropy_gap_threshold",
            0.0,
        ))
        if (
            not math.isfinite(entropy_gap_threshold)
            or entropy_gap_threshold < 0
        ):
            raise ValueError(
                "opsd_entropy_gap_threshold must be finite and non-negative."
            )
        entropy_gate_temperature = float(getattr(
            self.args,
            "opsd_entropy_gate_temperature",
            0.1,
        ))
        if (
            not math.isfinite(entropy_gate_temperature)
            or entropy_gate_temperature <= 0
        ):
            raise ValueError(
                "opsd_entropy_gate_temperature must be finite and positive."
            )
        entropy_min_mass = float(getattr(
            self.args,
            "opsd_entropy_gate_min_teacher_topk_mass",
            0.0,
        ))
        if (
            not math.isfinite(entropy_min_mass)
            or entropy_min_mass < 0
            or entropy_min_mass > 1
        ):
            raise ValueError(
                "opsd_entropy_gate_min_teacher_topk_mass must be in [0, 1]."
            )
        entropy_loss_normalization = (
            _normalize_opsd_entropy_gate_loss_normalization(getattr(
                self.args,
                "opsd_entropy_gate_loss_normalization",
                "active_tokens",
            ))
        )
        if (
            entropy_gate_mode != "none"
            and float(self.args.opsd_token_weight) == 0
        ):
            raise ValueError(
                "Entropy-gap token gating requires a non-zero "
                "opsd_token_weight."
            )
        self.args.opsd_entropy_gate_mode = entropy_gate_mode
        self.args.opsd_entropy_gap_threshold = entropy_gap_threshold
        self.args.opsd_entropy_gate_temperature = entropy_gate_temperature
        self.args.opsd_entropy_gate_min_teacher_topk_mass = entropy_min_mass
        self.args.opsd_entropy_gate_loss_normalization = (
            entropy_loss_normalization
        )
        if entropy_gate_mode != "none" and self.accelerator.is_main_process:
            print(
                "Token entropy-gap gating enabled: "
                f"mode={entropy_gate_mode}, "
                f"threshold={entropy_gap_threshold}, "
                f"temperature={entropy_gate_temperature}, "
                f"min_teacher_topk_mass={entropy_min_mass}, "
                f"loss_normalization={entropy_loss_normalization}."
            )

        target_weight = float(self.args.opsd_target_weight)
        if not math.isfinite(target_weight) or target_weight < 0:
            raise ValueError(
                "opsd_target_weight must be finite and non-negative."
            )
        target_loss_type = _normalize_opsd_target_loss_type(
            getattr(self.args, "opsd_target_loss_type", "infonce")
        )
        self.args.opsd_target_loss_type = target_loss_type
        directau_weight = float(getattr(
            self.args,
            "opsd_directau_weight",
            0.0,
        ))
        if not math.isfinite(directau_weight) or directau_weight < 0:
            raise ValueError(
                "opsd_directau_weight must be finite and non-negative."
            )
        if (
            directau_weight != 0
            and target_weight != 0
            and target_loss_type == "privileged_directau"
        ):
            raise ValueError(
                "Do not enable opsd_directau_weight together with the legacy "
                "privileged_directau target-loss type; use an InfoNCE or "
                "full-softmax target loss for additive DirectAU."
            )
        self.args.opsd_directau_weight = directau_weight

        teacher_checkpoint = getattr(
            self.args,
            "opsd_teacher_checkpoint",
            None,
        )
        if teacher_checkpoint is not None:
            teacher_checkpoint = str(teacher_checkpoint).strip()
            if not teacher_checkpoint:
                raise ValueError("opsd_teacher_checkpoint must not be empty.")
            self.args.opsd_teacher_checkpoint = teacher_checkpoint

        self._opsd_teacher_update_mode = _resolve_opsd_teacher_update_mode(
            mode=self.args.opsd_teacher_update_mode,
            teacher_checkpoint=teacher_checkpoint,
            token_weight=float(self.args.opsd_token_weight),
            rank_weight=self._effective_rank_weight,
            target_weight=target_weight,
            target_loss_type=target_loss_type,
            directau_weight=directau_weight,
        )
        self.args.opsd_teacher_update_mode = self._opsd_teacher_update_mode
        self.args.opsd_frozen_teacher = (
            self._opsd_teacher_update_mode in {"fixed", "lagged"}
        )
        teacher_sync_steps = int(self.args.opsd_teacher_sync_steps)
        if self._opsd_teacher_update_mode == "lagged" and teacher_sync_steps <= 0:
            raise ValueError(
                "opsd_teacher_sync_steps must be positive in lagged mode."
            )
        self.args.opsd_teacher_sync_steps = teacher_sync_steps

        teacher_gate_mode = _normalize_opsd_teacher_gate_mode(getattr(
            self.args,
            "opsd_teacher_gate_mode",
            "hard",
        ))
        teacher_gate_temperature = float(getattr(
            self.args,
            "opsd_teacher_gate_temperature",
            0.1,
        ))
        teacher_gate_power = float(getattr(
            self.args,
            "opsd_teacher_gate_power",
            1.0,
        ))
        if (
            not math.isfinite(teacher_gate_temperature)
            or teacher_gate_temperature <= 0
        ):
            raise ValueError(
                "opsd_teacher_gate_temperature must be finite and positive."
            )
        if not math.isfinite(teacher_gate_power) or teacher_gate_power <= 0:
            raise ValueError(
                "opsd_teacher_gate_power must be finite and positive."
            )
        self.args.opsd_teacher_gate_mode = teacher_gate_mode
        self.args.opsd_teacher_gate_temperature = teacher_gate_temperature
        self.args.opsd_teacher_gate_power = teacher_gate_power
        if (
            teacher_gate_mode == "advantage_weighted"
            and bool(self.args.opsd_teacher_better_only)
            and self.accelerator.is_main_process
        ):
            print(
                "Recommendation-advantage weighted OPSD enabled: "
                f"temperature={teacher_gate_temperature}, "
                f"power={teacher_gate_power}."
            )

        sasrec_teacher_gate_mode = _normalize_sasrec_teacher_gate_mode(
            getattr(self.args, "sasrec_teacher_gate_mode", "hard")
        )
        sasrec_teacher_reward_mode = _normalize_sasrec_teacher_reward_mode(
            getattr(
                self.args,
                "sasrec_teacher_reward_mode",
                "normalized_log_rank",
            )
        )
        sasrec_teacher_gate_temperature = float(getattr(
            self.args,
            "sasrec_teacher_gate_temperature",
            0.1,
        ))
        sasrec_teacher_gate_power = float(getattr(
            self.args,
            "sasrec_teacher_gate_power",
            1.0,
        ))
        if (
            not math.isfinite(sasrec_teacher_gate_temperature)
            or sasrec_teacher_gate_temperature <= 0
        ):
            raise ValueError(
                "sasrec_teacher_gate_temperature must be finite and positive."
            )
        if (
            not math.isfinite(sasrec_teacher_gate_power)
            or sasrec_teacher_gate_power <= 0
        ):
            raise ValueError(
                "sasrec_teacher_gate_power must be finite and positive."
            )
        self.args.sasrec_teacher_gate_mode = sasrec_teacher_gate_mode
        self.args.sasrec_teacher_reward_mode = sasrec_teacher_reward_mode
        self.args.sasrec_teacher_gate_temperature = (
            sasrec_teacher_gate_temperature
        )
        self.args.sasrec_teacher_gate_power = sasrec_teacher_gate_power
        if (
            sasrec_teacher_gate_mode == "advantage_weighted"
            and bool(getattr(
                self.args,
                "sasrec_teacher_better_only",
                True,
            ))
            and float(getattr(self.args, "sasrec_rank_weight", 0.0)) != 0
            and self.accelerator.is_main_process
        ):
            print(
                "SASRec recommendation-advantage soft gate enabled: "
                f"reward={sasrec_teacher_reward_mode}, "
                f"temperature={sasrec_teacher_gate_temperature}, "
                f"power={sasrec_teacher_gate_power}."
            )

        rationale_geometry_weight = float(getattr(
            self.args,
            "opsd_rationale_geometry_weight",
            0.0,
        ))
        rationale_geometry_loss_type = (
            _normalize_opsd_rationale_geometry_loss_type(getattr(
                self.args,
                "opsd_rationale_geometry_loss_type",
                "relational_kl",
            ))
        )
        rationale_reason_temperature = float(getattr(
            self.args,
            "opsd_rationale_reason_temperature",
            0.2,
        ))
        rationale_item_temperature = float(getattr(
            self.args,
            "opsd_rationale_item_temperature",
            0.2,
        ))
        if (
            not math.isfinite(rationale_geometry_weight)
            or rationale_geometry_weight < 0
        ):
            raise ValueError(
                "opsd_rationale_geometry_weight must be finite and "
                "non-negative."
            )
        if (
            not math.isfinite(rationale_reason_temperature)
            or rationale_reason_temperature <= 0
        ):
            raise ValueError(
                "opsd_rationale_reason_temperature must be finite and "
                "positive."
            )
        if (
            not math.isfinite(rationale_item_temperature)
            or rationale_item_temperature <= 0
        ):
            raise ValueError(
                "opsd_rationale_item_temperature must be finite and positive."
            )
        self.args.opsd_rationale_geometry_weight = rationale_geometry_weight
        self.args.opsd_rationale_geometry_loss_type = (
            rationale_geometry_loss_type
        )
        self.args.opsd_rationale_reason_temperature = (
            rationale_reason_temperature
        )
        self.args.opsd_rationale_item_temperature = (
            rationale_item_temperature
        )
        if rationale_geometry_weight != 0 and self.accelerator.is_main_process:
            print(
                "Headless item-geometry rationale loss enabled: "
                f"weight={rationale_geometry_weight}, "
                f"loss_type={rationale_geometry_loss_type}, "
                f"reason_temperature={rationale_reason_temperature}, "
                f"item_temperature={rationale_item_temperature}."
            )

        gradient_coordination_mode = (
            _normalize_opsd_gradient_coordination_mode(getattr(
                self.args,
                "opsd_gradient_coordination_mode",
                "sum",
            ))
        )
        gradient_coordination_epsilon = float(getattr(
            self.args,
            "opsd_gradient_coordination_epsilon",
            1e-12,
        ))
        if (
            not math.isfinite(gradient_coordination_epsilon)
            or gradient_coordination_epsilon <= 0
        ):
            raise ValueError(
                "opsd_gradient_coordination_epsilon must be finite and "
                "positive."
            )
        if gradient_coordination_mode != "sum":
            if float(self.args.opsd_token_weight) == 0:
                raise ValueError(
                    "Gradient coordination requires a non-zero "
                    "opsd_token_weight."
                )
            recommendation_weight_sum = sum(abs(float(weight)) for weight in (
                self.args.opsd_target_weight,
                getattr(self.args, "opsd_na_weight", 0.0),
                directau_weight,
                rationale_geometry_weight,
                self._effective_rank_weight,
                getattr(self.args, "lightgcn_rank_weight", 0.0),
                getattr(self.args, "sasrec_rank_weight", 0.0),
            ))
            if recommendation_weight_sum == 0:
                raise ValueError(
                    "Gradient coordination requires at least one non-zero "
                    "recommendation-space loss weight."
                )
        self.args.opsd_gradient_coordination_mode = (
            gradient_coordination_mode
        )
        self.args.opsd_gradient_coordination_epsilon = (
            gradient_coordination_epsilon
        )
        if (
            gradient_coordination_mode != "sum"
            and self.accelerator.is_main_process
        ):
            print(
                "OPSD gradient coordination enabled: "
                f"mode={gradient_coordination_mode}, "
                f"epsilon={gradient_coordination_epsilon}."
            )

        token_credit_mode = _normalize_opsd_token_credit_mode(
            getattr(self.args, "opsd_token_credit_mode", "uniform")
        )
        token_credit_temperature = float(getattr(
            self.args,
            "opsd_token_credit_temperature",
            0.2,
        ))
        token_credit_mix = float(getattr(
            self.args,
            "opsd_token_credit_mix",
            1.0,
        ))
        if (
            not math.isfinite(token_credit_temperature)
            or token_credit_temperature <= 0
        ):
            raise ValueError(
                "opsd_token_credit_temperature must be finite and positive."
            )
        if (
            not math.isfinite(token_credit_mix)
            or not 0.0 <= token_credit_mix <= 1.0
        ):
            raise ValueError("opsd_token_credit_mix must be in [0, 1].")
        if (
            token_credit_mode != "uniform"
            and float(self.args.opsd_token_weight) == 0
        ):
            raise ValueError(
                "Recommendation-aware token credit requires a non-zero "
                "opsd_token_weight."
            )
        self.args.opsd_token_credit_mode = token_credit_mode
        self.args.opsd_token_credit_temperature = token_credit_temperature
        self.args.opsd_token_credit_mix = token_credit_mix
        if (
            token_credit_mode != "uniform"
            and self.accelerator.is_main_process
        ):
            print(
                "Recommendation-aware token OPSD enabled: "
                f"mode={token_credit_mode}, "
                f"temperature={token_credit_temperature}, "
                f"mix={token_credit_mix}."
            )

        rank_loss_type = _normalize_opsd_rank_loss_type(getattr(
            self.args,
            "opsd_rank_loss_type",
            "distribution_kl",
        ))
        self.args.opsd_rank_loss_type = rank_loss_type
        if (
            rank_loss_type == "geometry_margin"
            and self._effective_rank_weight != 0
            and self.accelerator.is_main_process
        ):
            print(
                "User-item geometry distillation enabled: "
                f"positive_margin_only={bool(getattr(self.args, 'opsd_geometry_positive_margin_only', True))}."
            )

        max_token_kl = getattr(self.args, "opsd_max_token_kl", None)
        if max_token_kl is not None:
            max_token_kl = float(max_token_kl)
            if not math.isfinite(max_token_kl) or max_token_kl <= 0:
                raise ValueError(
                    "opsd_max_token_kl must be finite and positive."
                )
            if float(self.args.opsd_token_weight) == 0:
                raise ValueError(
                    "opsd_max_token_kl requires a non-zero "
                    "opsd_token_weight."
                )
            token_divergence = str(
                self.args.opsd_loss_type
            ).strip().lower()
            if token_divergence not in {"kl", "reverse_kl"}:
                raise ValueError(
                    "opsd_max_token_kl is defined for forward or reverse KL "
                    "distillation; set opsd_loss_type=kl or reverse_kl."
                )
            self.args.opsd_max_token_kl = max_token_kl

        _validate_frozen_teacher_objectives(
            frozen_teacher=bool(self.args.opsd_frozen_teacher),
            token_weight=float(self.args.opsd_token_weight),
            rank_weight=self._effective_rank_weight,
            privileged_target_enabled=(
                target_weight != 0
                and target_loss_type == "privileged_directau"
            ),
            independent_directau_enabled=directau_weight != 0,
        )

        if bool(self.args.opsd_frozen_teacher):

            # ``no_grad`` on the live student only prevents gradients during
            # one forward pass; it does not stop that same model from changing
            # after the optimizer step. Keep an independent snapshot so the
            # teacher remains fixed throughout OPSD.  Without an external
            # checkpoint this preserves the original snapshot behaviour.
            (
                self._opsd_frozen_teacher_model,
                normalized_teacher_checkpoint,
            ) = _build_frozen_opsd_teacher_model(
                self.model,
                teacher_checkpoint,
            )
            if normalized_teacher_checkpoint is not None:
                self.args.opsd_teacher_checkpoint = (
                    normalized_teacher_checkpoint
                )

            if self._opsd_teacher_update_mode == "lagged":
                student_model = self.accelerator.unwrap_model(self.model)
                (
                    self._opsd_lagged_parameter_names,
                    self._opsd_teacher_last_update_norm,
                ) = _copy_trainable_student_parameters(
                    student_model,
                    self._opsd_frozen_teacher_model,
                )

            if self.accelerator.is_main_process:
                if self._opsd_teacher_update_mode == "lagged":
                    print(
                        "Initialized a lagged OPSD self-teacher from the "
                        "starting student; synchronizing every "
                        f"{teacher_sync_steps} optimizer steps."
                    )
                elif teacher_checkpoint is None:
                    print(
                        "Initialized an independent fixed OPSD teacher "
                        "from the starting student."
                    )
                else:
                    print(
                        "Loaded an independent frozen OPSD privileged teacher "
                        "adapter from "
                        f"{self.args.opsd_teacher_checkpoint}."
                    )
        alignment_mode = _normalize_opsd_privileged_alignment_mode(getattr(
            self.args,
            "opsd_privileged_alignment_mode",
            "cross_user_item",
        ))
        privileged_alignment_weight = float(getattr(
            self.args,
            "opsd_privileged_alignment_weight",
            1.0,
        ))
        directau_uniform_weight = float(getattr(
            self.args,
            "opsd_directau_uniform_weight",
            1.0,
        ))
        directau_uniform_temperature = float(getattr(
            self.args,
            "opsd_directau_uniform_temperature",
            2.0,
        ))
        for name, value in (
            ("opsd_privileged_alignment_weight", privileged_alignment_weight),
            ("opsd_directau_uniform_weight", directau_uniform_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if (
            not math.isfinite(directau_uniform_temperature)
            or directau_uniform_temperature <= 0
        ):
            raise ValueError(
                "opsd_directau_uniform_temperature must be finite and positive."
            )
        self.args.opsd_privileged_alignment_mode = alignment_mode
        self.args.opsd_privileged_alignment_weight = (
            privileged_alignment_weight
        )
        self.args.opsd_directau_uniform_weight = directau_uniform_weight
        self.args.opsd_directau_uniform_temperature = (
            directau_uniform_temperature
        )
        if self.privileged_directau_enabled and self.accelerator.is_main_process:
            objective_kind = (
                "independent auxiliary"
                if directau_weight != 0
                else "legacy target"
            )
            directau_outer_weight = (
                directau_weight if directau_weight != 0 else target_weight
            )
            print(
                "Privileged DirectAU objective enabled: "
                f"kind={objective_kind}, outer_weight={directau_outer_weight}, "
                f"alignment_mode={alignment_mode}, "
                f"privileged_weight={privileged_alignment_weight}, "
                f"uniform_weight={directau_uniform_weight}, "
                f"uniform_temperature={directau_uniform_temperature}."
            )
        na_weight = float(getattr(self.args, "opsd_na_weight", 0.0))
        if not math.isfinite(na_weight) or na_weight < 0:
            raise ValueError(
                "opsd_na_weight must be finite and non-negative."
            )
        na_temperature = float(
            getattr(self.args, "opsd_na_temperature", 0.2)
        )
        if not math.isfinite(na_temperature) or na_temperature <= 0:
            raise ValueError(
                "opsd_na_temperature must be finite and positive."
            )
        if na_weight != 0 and (
            getattr(self.args, "similarity_type", "dot") != "dot"
            or not bool(
                getattr(self.args, "similarity_normalization", True)
            )
        ):
            raise ValueError(
                "LightCCF NA requires similarity_type='dot' with "
                "similarity_normalization=True so its reused item scores "
                "are cosine similarities."
            )

        item_bank_refresh_steps = int(
            self.args.opsd_item_bank_refresh_steps
        )
        if item_bank_refresh_steps < -1:
            raise ValueError(
                "opsd_item_bank_refresh_steps must be -1, 0, or positive."
            )
        if item_bank_refresh_steps >= 0 and self.accelerator.is_main_process:
            refresh_description = (
                "at evaluation only"
                if item_bank_refresh_steps == 0
                else f"every {item_bank_refresh_steps} optimizer steps"
            )
            print(
                "OPSD item bank uses version-consistent full snapshots; "
                f"refreshing {refresh_description}."
            )

        privileged_context = str(
            self.args.opsd_privileged_context
        ).strip().lower()
        if privileged_context not in {
            "target",
            "future",
            "target_hard_negatives",
        }:
            raise ValueError(
                "opsd_privileged_context must be one of: target, future, "
                "target_hard_negatives."
            )
        self.args.opsd_privileged_context = privileged_context
        hard_negative_count = int(getattr(
            self.args,
            "opsd_hard_negative_count",
            3,
        ))
        if hard_negative_count <= 0:
            raise ValueError("opsd_hard_negative_count must be positive.")
        self.args.opsd_hard_negative_count = hard_negative_count
        self.args.opsd_hard_negative_exclude_train_positives = bool(getattr(
            self.args,
            "opsd_hard_negative_exclude_train_positives",
            True,
        ))

        self._opsd_future_items_by_train_data_id = {}
        if privileged_context == "future" and self._privileged_opsd_enabled:
            required_columns = {
                "train_data_id",
                "user_id",
                "seq_labels",
                "item_title",
                "timestamp",
            }
            missing_columns = required_columns.difference(
                self.train_dataset.column_names
            )
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(
                    "OPSD future context requires these train columns: "
                    f"{missing}."
                )

            selected_columns = required_columns.union(
                {"interaction_id", "rating"}.intersection(
                    self.train_dataset.column_names
                )
            )
            column_values = {
                name: self.train_dataset[name]
                for name in selected_columns
            }
            records = (
                {
                    name: values[row_index]
                    for name, values in column_values.items()
                }
                for row_index in range(len(self.train_dataset))
            )
            self._opsd_future_items_by_train_data_id = (
                _build_train_future_item_map(
                    records,
                    max_items=int(self.args.dataset_window_size),
                )
            )
            if self.accelerator.is_main_process:
                covered_rows = len(
                    self._opsd_future_items_by_train_data_id
                )
                print(
                    "Built train-only OPSD future context for "
                    f"{covered_rows}/{len(self.train_dataset)} rows "
                    f"(up to {int(self.args.dataset_window_size)} items per row)."
                )

        self._opsd_train_positive_item_ids_by_user_id = {}
        if (
            privileged_context == "target_hard_negatives"
            and self._privileged_opsd_enabled
            and self.args.opsd_hard_negative_exclude_train_positives
        ):
            required_columns = {"user_id", "seq_labels"}
            missing_columns = required_columns.difference(
                self.train_dataset.column_names
            )
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(
                    "HN-OPD train-positive filtering requires these train "
                    f"columns: {missing}."
                )
            records = (
                {
                    "user_id": user_id,
                    "seq_labels": item_id,
                }
                for user_id, item_id in zip(
                    self.train_dataset["user_id"],
                    self.train_dataset["seq_labels"],
                )
            )
            self._opsd_train_positive_item_ids_by_user_id = (
                _build_train_positive_item_map(records)
            )
            if self.accelerator.is_main_process:
                print(
                    "HN-OPD will exclude training positives for "
                    f"{len(self._opsd_train_positive_item_ids_by_user_id)} users."
                )

        if (
            privileged_context == "target_hard_negatives"
            and self._privileged_opsd_enabled
            and self.accelerator.is_main_process
        ):
            print(
                "Hard-negative privileged OPSD enabled: "
                f"student_topm={hard_negative_count}, "
                "source=online_rollout_full_catalog, "
                "exclude_train_positives="
                f"{self.args.opsd_hard_negative_exclude_train_positives}."
            )

        self.lightgcn_bank: Optional[LightGCNEmbeddingBank] = None
        if self.lightgcn_enabled:
            checkpoint = self.args.lightgcn_checkpoint
            if not checkpoint:
                raise ValueError(
                    "lightgcn_rank_weight is non-zero but no "
                    "lightgcn_checkpoint was provided."
                )
            self.lightgcn_bank = LightGCNEmbeddingBank.from_checkpoint(
                checkpoint,
                expected_num_items=len(self.item_dataset),
            ).to(self.accelerator.device)
            if self.accelerator.is_main_process:
                print(
                    "Loaded frozen LightGCN embeddings from "
                    f"{self.lightgcn_bank.source_path}"
                )

        self.sasrec_teacher: Optional[SASRecTeacher] = None
        sasrec_weight = float(getattr(self.args, "sasrec_rank_weight", 0.0))
        if not math.isfinite(sasrec_weight) or sasrec_weight < 0:
            raise ValueError(
                "sasrec_rank_weight must be finite and non-negative."
            )
        if self.sasrec_enabled:
            checkpoint = getattr(self.args, "sasrec_checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "sasrec_rank_weight is non-zero but no "
                    "sasrec_checkpoint was provided."
                )
            self.sasrec_teacher = SASRecTeacher.from_checkpoint(
                checkpoint,
                expected_num_items=len(self.item_dataset),
            ).to(self.accelerator.device)
            if self.accelerator.is_main_process:
                print(
                    "Loaded frozen SASRec teacher from "
                    f"{self.sasrec_teacher.source_path}"
                )

        self.small_teacher_moe: Optional[SmallTeacherMoE] = None
        small_moe_weight = float(getattr(
            self.args,
            "opsd_small_moe_weight",
            0.0,
        ))
        if not math.isfinite(small_moe_weight) or small_moe_weight < 0:
            raise ValueError(
                "opsd_small_moe_weight must be finite and non-negative."
            )
        self.args.opsd_small_moe_weight = small_moe_weight
        self.args.opsd_small_moe_router_mode = normalize_small_moe_router_mode(
            getattr(self.args, "opsd_small_moe_router_mode", "soft_mixture")
        )
        self.args.opsd_small_moe_advantage_mode = (
            normalize_small_moe_advantage_mode(
                getattr(
                    self.args,
                    "opsd_small_moe_advantage_mode",
                    "target_rank",
                )
            )
        )
        small_moe_topk = int(getattr(self.args, "opsd_small_moe_topk", 32))
        if small_moe_topk < 0:
            raise ValueError(
                "opsd_small_moe_topk must be non-negative; use 0 for full catalog."
            )
        self.args.opsd_small_moe_topk = small_moe_topk
        positive_small_moe_arguments = (
            "opsd_small_moe_student_temperature",
            "opsd_small_moe_sasrec_temperature",
            "opsd_small_moe_lightgcn_temperature",
            "opsd_small_moe_fmlprec_temperature",
            "opsd_small_moe_ease_temperature",
            "opsd_small_moe_gate_temperature",
            "opsd_small_moe_outer_temperature",
        )
        for argument_name in positive_small_moe_arguments:
            value = float(getattr(self.args, argument_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{argument_name} must be finite and positive."
                )
            setattr(self.args, argument_name, value)
        quality_power = float(getattr(
            self.args,
            "opsd_small_moe_quality_power",
            1.0,
        ))
        if not math.isfinite(quality_power) or quality_power < 0:
            raise ValueError(
                "opsd_small_moe_quality_power must be finite and non-negative."
            )
        self.args.opsd_small_moe_quality_power = quality_power
        if self.small_moe_enabled:
            checkpoint_arguments = {
                "sasrec_checkpoint": getattr(
                    self.args,
                    "small_moe_sasrec_checkpoint",
                    None,
                ),
                "lightgcn_checkpoint": getattr(
                    self.args,
                    "small_moe_lightgcn_checkpoint",
                    None,
                ),
                "fmlprec_checkpoint": getattr(
                    self.args,
                    "small_moe_fmlprec_checkpoint",
                    None,
                ),
                "ease_checkpoint": getattr(
                    self.args,
                    "small_moe_ease_checkpoint",
                    None,
                ),
            }
            missing = [
                name for name, checkpoint in checkpoint_arguments.items()
                if not checkpoint
            ]
            if missing:
                raise ValueError(
                    "opsd_small_moe_weight is non-zero but these independent "
                    f"MoE checkpoints are missing: {', '.join(missing)}."
                )
            self.small_teacher_moe = SmallTeacherMoE.from_checkpoints(
                expected_num_items=len(self.item_dataset),
                **checkpoint_arguments,
            ).to(self.accelerator.device)
            if self.accelerator.is_main_process:
                quality_summary = ", ".join(
                    f"{name}={quality:.6f}"
                    for name, quality in (
                        self.small_teacher_moe.validation_qualities.items()
                    )
                )
                print(
                    "Loaded independent four-expert MoE: "
                    "SASRec + LightGCN + FMLP-Rec + EASE; "
                    f"weight={small_moe_weight}, topk={small_moe_topk}, "
                    f"router={self.args.opsd_small_moe_router_mode}, "
                    f"advantage={self.args.opsd_small_moe_advantage_mode}, "
                    f"validation_ndcg=({quality_summary})."
                )

    def _make_mini_batch_divisible(self, macro_batch_size: int) -> None:
        """Avoid the parent trainer's extra iterator on a remainder batch.

        The original ``training_step`` computes the number of mini batches as
        ``macro // mini + macro % mini``.  For a non-zero remainder that is
        larger than ``ceil(macro / mini)``.  Choosing a nearby divisor here
        keeps the original training step reusable without changing its file.
        """

        current = max(1, int(self.args.mini_batch_size))
        if macro_batch_size % current == 0:
            return
        self.args.mini_batch_size = max(
            1,
            math.gcd(macro_batch_size, current),
        )

    @property
    def _use_consistent_item_bank(self) -> bool:
        return int(
            getattr(self.args, "opsd_item_bank_refresh_steps", -1)
        ) >= 0

    def _maybe_update_frozen_teacher(self, model) -> bool:
        """Synchronize a lagged self-teacher once per optimizer interval."""

        if self._opsd_teacher_update_mode != "lagged":
            return False
        teacher_model = self._opsd_frozen_teacher_model
        if teacher_model is None:
            raise RuntimeError("Lagged OPSD mode requires an independent teacher.")
        current_step = int(self.state.global_step)
        sync_steps = int(self.args.opsd_teacher_sync_steps)
        if current_step <= self._opsd_teacher_last_sync_step:
            return False
        if current_step - self._opsd_teacher_last_sync_step < sync_steps:
            return False

        wait_for_everyone = getattr(
            self.accelerator,
            "wait_for_everyone",
            None,
        )
        if callable(wait_for_everyone):
            wait_for_everyone()
        student_model = self.accelerator.unwrap_model(model)
        (
            self._opsd_lagged_parameter_names,
            self._opsd_teacher_last_update_norm,
        ) = _copy_trainable_student_parameters(
            student_model,
            teacher_model,
            parameter_names=self._opsd_lagged_parameter_names,
        )
        self._opsd_teacher_last_sync_step = current_step
        self._opsd_teacher_sync_count += 1

        # User and item representations must describe the same model version.
        # Force both banks to rebuild before any reward/rank comparison.
        self.item_hs = None
        self._opsd_item_bank_last_refresh_step = None
        self._opsd_frozen_teacher_item_hs = None
        if callable(wait_for_everyone):
            wait_for_everyone()
        if self.accelerator.is_main_process:
            print(
                "Synchronized lagged OPSD teacher from student at optimizer "
                f"step {current_step} (adapter L2 update "
                f"{self._opsd_teacher_last_update_norm:.6f})."
            )
        return True

    def _opsd_teacher_state_payload(self):
        if self._opsd_teacher_update_mode != "lagged":
            return None
        return {
            "version": 1,
            "mode": "lagged",
            "last_sync_step": int(self._opsd_teacher_last_sync_step),
            "sync_count": int(self._opsd_teacher_sync_count),
            "last_update_norm": float(self._opsd_teacher_last_update_norm),
            "parameter_names": tuple(self._opsd_lagged_parameter_names),
            "teacher_state": _snapshot_named_parameters(
                self._opsd_frozen_teacher_model,
                self._opsd_lagged_parameter_names,
            ),
        }

    def _restore_opsd_teacher_state(self, checkpoint_dir) -> bool:
        state_path = Path(checkpoint_dir) / _OPSD_TEACHER_STATE_FILENAME
        if not state_path.is_file():
            return False
        payload = _load_tensor_sidecar(state_path)
        if payload.get("mode") != self._opsd_teacher_update_mode:
            raise ValueError(
                "Saved OPSD teacher mode does not match this run: "
                f"{payload.get('mode')!r} != "
                f"{self._opsd_teacher_update_mode!r}."
            )
        _load_named_parameters(
            self._opsd_frozen_teacher_model,
            payload["teacher_state"],
        )
        self._opsd_lagged_parameter_names = tuple(
            payload["parameter_names"]
        )
        self._opsd_teacher_last_sync_step = int(payload["last_sync_step"])
        self._opsd_teacher_sync_count = int(payload["sync_count"])
        self._opsd_teacher_last_update_norm = float(
            payload.get("last_update_norm", 0.0)
        )
        self._opsd_frozen_teacher_item_hs = None
        return True

    def save_model(self, output_dir=None, _internal_call=False):
        result = super().save_model(
            output_dir=output_dir,
            _internal_call=_internal_call,
        )
        payload = self._opsd_teacher_state_payload()
        if payload is None or not bool(getattr(self.args, "should_save", True)):
            return result
        target_dir = Path(output_dir or self.args.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        state_path = target_dir / _OPSD_TEACHER_STATE_FILENAME
        temporary_path = target_dir / f"{_OPSD_TEACHER_STATE_FILENAME}.tmp"
        torch.save(payload, temporary_path)
        temporary_path.replace(state_path)
        return result

    def _load_from_checkpoint(
            self,
            resume_from_checkpoint,
            model=None,
            **kwargs,
    ):
        result = super()._load_from_checkpoint(
            resume_from_checkpoint,
            model=model,
            **kwargs,
        )
        self._restore_opsd_teacher_state(resume_from_checkpoint)
        return result

    def _load_best_model(self):
        """Keep a stateful teacher aligned with the restored best student."""

        result = super()._load_best_model()
        best_checkpoint = getattr(
            self.state,
            "best_model_checkpoint",
            None,
        )
        if best_checkpoint is not None:
            self._restore_opsd_teacher_state(best_checkpoint)
        return result

    def _generate_item_embeddings(self, *args, **kwargs):
        """Track the model step represented by each full item-bank snapshot."""

        result = super()._generate_item_embeddings(*args, **kwargs)
        state = getattr(self, "state", None)
        self._opsd_item_bank_last_refresh_step = int(
            getattr(state, "global_step", 0)
        )
        return result

    def _generate_frozen_teacher_item_embeddings(self):
        """Build one catalog snapshot in the frozen teacher's own space.

        The parent embedding routine writes its result to ``self.item_hs``.
        Preserve the student's bank around that call so teacher construction
        cannot replace or refresh the trainable student's catalog snapshot.
        """

        cached_item_hs = getattr(
            self,
            "_opsd_frozen_teacher_item_hs",
            None,
        )
        if cached_item_hs is not None:
            return cached_item_hs

        teacher_model = getattr(
            self,
            "_opsd_frozen_teacher_model",
            None,
        )
        if teacher_model is None:
            raise RuntimeError(
                "A frozen OPSD teacher item bank requires a frozen teacher."
            )

        student_item_hs = self.item_hs
        try:
            # Bypass this class's student-bank refresh tracker. The isolated
            # snapshot is generated once per teacher version; lagged and
            # anchored lifecycle updates explicitly invalidate it.
            super()._generate_item_embeddings(teacher_model)
            if self.item_hs is None:
                raise RuntimeError(
                    "Frozen OPSD teacher item-bank generation returned no "
                    "embeddings."
                )
            self._opsd_frozen_teacher_item_hs = self.item_hs.detach()
        finally:
            self.item_hs = student_item_hs

        if self.accelerator.is_main_process:
            print(
                "Built an isolated OPSD teacher item bank "
                f"with {self._opsd_frozen_teacher_item_hs.shape[0]} items."
            )
        return self._opsd_frozen_teacher_item_hs

    def _get_teacher_item_bank_for_scoring(self, student_item_hs):
        """Return an item bank aligned with the active privileged teacher."""

        if getattr(self, "_opsd_frozen_teacher_model", None) is None:
            # The non-frozen privileged teacher is the live student model, so
            # both views intentionally share the same current item space.
            return student_item_hs

        teacher_item_hs = getattr(
            self,
            "_opsd_frozen_teacher_item_hs",
            None,
        )
        if teacher_item_hs is None:
            raise RuntimeError(
                "Frozen OPSD teacher ranking requires its isolated item bank."
            )
        if student_item_hs.ndim != 2 or teacher_item_hs.ndim != 2:
            raise RuntimeError(
                "Student and frozen-teacher item banks must be matrices."
            )
        if student_item_hs.shape[0] != teacher_item_hs.shape[0]:
            raise RuntimeError(
                "Student and frozen-teacher item banks have different "
                "catalog sizes."
            )
        return teacher_item_hs.detach()

    @property
    def _effective_rank_weight(self) -> float:
        """Prefer the new rank weight while accepting old launch commands."""

        rank_weight = float(self.args.opsd_rank_weight)
        if rank_weight != 0:
            return rank_weight
        return float(self.args.opsd_embed_weight)

    @property
    def lightgcn_enabled(self) -> bool:
        return float(self.args.lightgcn_rank_weight) != 0

    @property
    def sasrec_enabled(self) -> bool:
        return float(getattr(self.args, "sasrec_rank_weight", 0.0)) != 0

    @property
    def small_moe_enabled(self) -> bool:
        return float(getattr(self.args, "opsd_small_moe_weight", 0.0)) != 0

    @property
    def target_recommendation_enabled(self) -> bool:
        return float(self.args.opsd_target_weight) != 0

    @property
    def target_infonce_enabled(self) -> bool:
        """Backward-compatible alias for the selectable target objective."""

        return self.target_recommendation_enabled

    @property
    def directau_enabled(self) -> bool:
        """Whether the independent additive DirectAU objective is active."""

        return float(getattr(self.args, "opsd_directau_weight", 0.0)) != 0

    @property
    def rationale_geometry_enabled(self) -> bool:
        """Whether target-item geometry structures rollout rationales."""

        return float(getattr(
            self.args,
            "opsd_rationale_geometry_weight",
            0.0,
        )) != 0

    @property
    def privileged_directau_enabled(self) -> bool:
        return (
            self.directau_enabled
            or (
                self.target_recommendation_enabled
                and _normalize_opsd_target_loss_type(getattr(
                    self.args,
                    "opsd_target_loss_type",
                    "infonce",
                )) == "privileged_directau"
            )
        )

    @property
    def token_credit_enabled(self) -> bool:
        return _normalize_opsd_token_credit_mode(getattr(
            self.args,
            "opsd_token_credit_mode",
            "uniform",
        )) != "uniform"

    @property
    def target_item_token_credit_enabled(self) -> bool:
        return _normalize_opsd_token_credit_mode(getattr(
            self.args,
            "opsd_token_credit_mode",
            "uniform",
        )) == "target_item_cosine"

    @property
    def hard_negative_context_enabled(self) -> bool:
        return (
            self._privileged_opsd_enabled
            and str(self.args.opsd_privileged_context).strip().lower()
            == "target_hard_negatives"
        )

    @property
    def na_enabled(self) -> bool:
        return float(getattr(self.args, "opsd_na_weight", 0.0)) != 0

    @property
    def _privileged_opsd_enabled(self) -> bool:
        return (
            float(self.args.opsd_token_weight) != 0
            or self._effective_rank_weight != 0
            or self.privileged_directau_enabled
        )

    @property
    def opsd_enabled(self) -> bool:
        return (
            self.target_recommendation_enabled
            or self.na_enabled
            or self._privileged_opsd_enabled
            or self.rationale_geometry_enabled
            or self.lightgcn_enabled
            or self.sasrec_enabled
            or self.small_moe_enabled
        )

    def compute_rec_score(
            self,
            model,
            inputs,
            similarity: Optional[torch.FloatTensor] = None,
            train_eval="train",
    ):
        """Skip OutcomeGRPO reward/advantage work that pure OPSD never consumes."""

        if train_eval == "train" and self.opsd_enabled:
            return {}
        return super().compute_rec_score(
            model=model,
            inputs=inputs,
            similarity=similarity,
            train_eval=train_eval,
        )

    def _build_privileged_prompt(
            self,
            example: Dict,
            hard_negative_ids=None,
    ) -> str:
        """Create teacher-only context while keeping the student prompt intact."""

        history_items = get_items_str(
            example,
            self.item_dataset,
            self.args.dataset_window_size,
            just_title=True,
        )
        base_prompt = (
            self.user_prompter.prompts["user_prompt"]
            + "\n"
            + history_items
        )

        if self.args.opsd_privileged_context == "target_hard_negatives":
            hard_negative_titles = []
            if hard_negative_ids is None:
                hard_negative_ids = ()
            for item_id in hard_negative_ids:
                item_id = int(item_id)
                if item_id <= 0:
                    continue
                if item_id >= len(self.item_dataset):
                    raise ValueError(
                        "A mined hard-negative item ID is outside the real "
                        "item catalog."
                    )
                item_record = self.item_dataset[item_id - 1]
                hard_negative_titles.append(
                    str(item_record.get("title") or f"Item {item_id}")
                )
            comparison_context = (
                _format_hard_negative_privileged_context(
                    target_title=example["item_title"],
                    hard_negative_titles=hard_negative_titles,
                )
            )
            return base_prompt + "\n\n" + comparison_context

        if self.args.opsd_privileged_context == "future":
            train_data_id = int(example["train_data_id"])
            future_items = self._opsd_future_items_by_train_data_id.get(
                train_data_id,
                (),
            )
            if not future_items:
                return base_prompt

            future_lines = []
            for _, title, rating in future_items:
                rating_suffix = ""
                if rating is not None:
                    rating_text = str(rating).strip()
                    if rating_text and rating_text.lower() not in {
                        "nan",
                        "none",
                    }:
                        rating_suffix = f" ({rating_text})"
                # Present teacher-only future positives in exactly the same
                # lightweight form as ordinary liked-history entries.  A small
                # model need not interpret meta-instructions about privileged
                # or future information; it simply receives a longer history.
                future_lines.append(f"[{title}]{rating_suffix}")

            return base_prompt + "\n" + "\n".join(future_lines)

        if getattr(self.args, "opsd_teacher_checkpoint", None):
            # Match stage-1 target-teacher training exactly.  The target is
            # exposed as one additional liked-history item without meta-words
            # such as "target", "future", or "privileged".
            target_title = str(example["item_title"])
            target_rating = example.get("rating")
            rating_suffix = ""
            if target_rating is not None:
                rating_text = str(target_rating).strip()
                if rating_text and rating_text.lower() not in {
                    "nan",
                    "none",
                }:
                    rating_suffix = f" ({rating_text})"
            return base_prompt + "\n" + f"[{target_title}]{rating_suffix}"

        target_item = get_item_info_str(
            example,
            self.item_dataset,
            just_title=False,
        )
        return (
            base_prompt
            + (
                "\n\nPrivileged training information:\n"
                "The following is the verified ground-truth recommendation:\n"
            )
            + target_item
            + (
                "\n\nUsing the user's history and this ground-truth "
                "recommendation, generate the ideal response to the original "
                "request. The analysis should support this recommendation and "
                "follow the requested answer format."
            )
        )

    def _encode_teacher_prompt(
            self,
            example: Dict,
            hard_negative_ids=None,
    ) -> list[int]:
        """Tokenize only the teacher prompt and its generation marker."""

        prompt = self._build_privileged_prompt(
            example,
            hard_negative_ids=hard_negative_ids,
        )
        prompt_text = self.processing_class.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processing_class(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        return encoded["input_ids"]

    def _build_teacher_batch(
            self,
            batch,
            student_batch,
            hard_negative_ids=None,
    ):
        """Build teacher inputs from the already-generated student tokens.

        The student completion is recovered from ``multi_user_labels``.  This
        avoids regenerating teacher text and guarantees exact token identity
        even when student and teacher prompts have different lengths.
        """

        train_data_ids = batch["train_data_id"]
        if isinstance(train_data_ids, torch.Tensor):
            train_data_ids = train_data_ids.detach().cpu().tolist()
        else:
            train_data_ids = list(train_data_ids)

        student_input_ids = student_batch["multi_user_input_ids"]
        student_labels = student_batch["multi_user_labels"]
        if student_input_ids.shape != student_labels.shape:
            raise ValueError("Student input ids and labels must have the same shape.")

        if len(train_data_ids) == 0:
            raise ValueError("Cannot build OPSD teacher inputs from an empty batch.")
        if student_input_ids.shape[0] % len(train_data_ids) != 0:
            raise ValueError(
                "The generated student batch is not divisible by the original batch."
            )

        num_samples = student_input_ids.shape[0] // len(train_data_ids)
        prompt_ids_by_example = None
        prompt_ids_by_row = None
        if self.hard_negative_context_enabled:
            if hard_negative_ids is None:
                raise ValueError(
                    "target_hard_negatives requires mined hard-negative IDs."
                )
            if (
                hard_negative_ids.ndim != 2
                or hard_negative_ids.shape[0] != student_input_ids.shape[0]
            ):
                raise ValueError(
                    "Hard-negative IDs must have shape [rollouts, topm]."
                )
            hard_negative_rows = hard_negative_ids.detach().cpu().tolist()
            prompt_ids_by_row = []
            for row, row_hard_negative_ids in enumerate(hard_negative_rows):
                example_index = row // num_samples
                example = dict(self.train_dataset[
                    int(train_data_ids[example_index])
                ])
                prompt_ids_by_row.append(self._encode_teacher_prompt(
                    example,
                    hard_negative_ids=row_hard_negative_ids,
                ))
        else:
            prompt_ids_by_example = []
            for train_data_id in train_data_ids:
                example = dict(self.train_dataset[int(train_data_id)])
                prompt_ids_by_example.append(
                    self._encode_teacher_prompt(example)
                )

        max_length = self.args.opsd_max_length
        if max_length is None:
            max_length = self.args.user_input_max_length
        if max_length <= 0:
            raise ValueError("opsd_max_length must be positive when provided.")

        teacher_examples = []
        truncated_prompts = 0
        for row in range(student_input_ids.shape[0]):
            example_index = row // num_samples
            completion_mask = student_labels[row] != -100
            completion_ids = student_input_ids[row][completion_mask]
            completion_ids = completion_ids.detach().cpu().tolist()
            if not completion_ids:
                raise ValueError(
                    "A student rollout has no valid completion tokens; "
                    "strict OPSD cannot align this example."
                )

            if prompt_ids_by_row is not None:
                prompt_ids = list(prompt_ids_by_row[row])
            else:
                prompt_ids = list(prompt_ids_by_example[example_index])
            available_prompt_length = max_length - len(completion_ids)
            if available_prompt_length <= 0:
                raise ValueError(
                    "The teacher prompt leaves no room for the student completion. "
                    "Increase opsd_max_length or reduce generation length."
                )
            if len(prompt_ids) > available_prompt_length:
                truncated_prompts += 1
                print(f"================Prompt truncated for example {example_index}, truncated length {len(prompt_ids) - available_prompt_length}==========================")
                # Preserve every student completion token.  The prompt is
                # truncated from the left only in this exceptional case.
                prompt_ids = prompt_ids[-available_prompt_length:]

            input_ids = prompt_ids + completion_ids
            teacher_examples.append({
                "opsd_teacher_input_ids": input_ids,
                "opsd_teacher_attention_mask": [1] * len(input_ids),
                "opsd_teacher_completion_range": (
                    len(prompt_ids),
                    len(input_ids),
                ),
            })

        teacher_batch = self.data_collator(teacher_examples)
        for key, value in teacher_batch.items():
            if isinstance(value, torch.Tensor):
                teacher_batch[key] = value.to(student_input_ids.device)
        self.store_metrics({
            "opsd_teacher_prompt_truncation_rate": (
                truncated_prompts / max(1, student_input_ids.shape[0])
            ),
        })
        return teacher_batch

    def _mine_online_student_hard_negatives(
            self,
            model,
            batch,
            student_batch,
            num_samples,
    ):
        """Mine rollout-conditioned student Top-M items without gradients."""

        if self.item_hs is None:
            raise RuntimeError(
                "HN-OPD requires an initialized student item bank."
            )
        input_ids = student_batch["multi_user_input_ids"]
        attention_mask = student_batch["multi_user_attention_mask"]
        macro_batch_size = input_ids.shape[0]
        mini_batch_size = max(1, int(self.args.mini_batch_size))

        score_chunks = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, macro_batch_size, mini_batch_size):
                    end = min(start + mini_batch_size, macro_batch_size)
                    sequence_hidden = self._call_model_for_opsd(
                        model,
                        input_ids[start:end],
                        attention_mask[start:end],
                    )
                    user_hidden = _last_non_padding_hidden(
                        sequence_hidden,
                        attention_mask[start:end],
                    )
                    score_chunks.append(self.similarity(
                        user_hidden,
                        self.item_hs.detach(),
                    ).detach())
        finally:
            model.train(was_training)
        student_scores = torch.cat(score_chunks, dim=0)

        known_positive_item_ids = None
        if self.args.opsd_hard_negative_exclude_train_positives:
            if "user_id" not in batch:
                raise KeyError(
                    "HN-OPD train-positive filtering requires user_id."
                )
            raw_user_ids = batch["user_id"]
            if isinstance(raw_user_ids, torch.Tensor):
                raw_user_ids = raw_user_ids.detach().cpu().tolist()
            else:
                raw_user_ids = list(raw_user_ids)
            positive_rows = [
                self._opsd_train_positive_item_ids_by_user_id.get(
                    str(user_id),
                    (),
                )
                for user_id in raw_user_ids
                for _ in range(num_samples)
            ]
            if len(positive_rows) != macro_batch_size:
                raise ValueError(
                    "HN-OPD user IDs cannot be aligned with student rollouts."
                )
            max_positive_count = max(
                (len(item_ids) for item_ids in positive_rows),
                default=0,
            )
            if max_positive_count > 0:
                known_positive_item_ids = torch.zeros(
                    (macro_batch_size, max_positive_count),
                    dtype=torch.long,
                    device=student_scores.device,
                )
                for row, item_ids in enumerate(positive_rows):
                    if item_ids:
                        known_positive_item_ids[
                            row,
                            :len(item_ids),
                        ] = torch.as_tensor(
                            item_ids,
                            dtype=torch.long,
                            device=student_scores.device,
                        )

        hard_negative_ids, metrics = _mine_student_hard_negative_ids(
            student_scores=student_scores,
            history_item_ids=student_batch["opsd_seq_input_ids"],
            target_item_ids=student_batch["opsd_seq_labels"],
            known_positive_item_ids=known_positive_item_ids,
            topm=int(self.args.opsd_hard_negative_count),
        )
        self.store_metrics({
            key: value.float().item()
            for key, value in metrics.items()
        })
        return hard_negative_ids

    def _generate_in_train(self, model, batch):
        """Reuse the original rollout, then append the dynamic teacher batch."""

        if self.opsd_enabled:
            self._maybe_update_frozen_teacher(model)

        needs_privileged_item_context = (
            self._effective_rank_weight != 0
            or self.privileged_directau_enabled
            or (
                float(self.args.opsd_token_weight) != 0
                and bool(self.args.opsd_teacher_better_only)
            )
        )
        needs_item_context = (
            self.target_recommendation_enabled
            or self.na_enabled
            or self.target_item_token_credit_enabled
            or self.hard_negative_context_enabled
            or needs_privileged_item_context
            or self.rationale_geometry_enabled
            or self.lightgcn_enabled
            or self.sasrec_enabled
            or self.small_moe_enabled
        )
        if self.opsd_enabled and needs_item_context:
            refresh_steps = int(self.args.opsd_item_bank_refresh_steps)
            current_step = int(self.state.global_step)
            last_refresh_step = self._opsd_item_bank_last_refresh_step
            needs_initial_bank = self.item_hs is None
            needs_periodic_refresh = (
                self._use_consistent_item_bank
                and refresh_steps > 0
                and self.item_hs is not None
                and (
                    last_refresh_step is None
                    or current_step - last_refresh_step >= refresh_steps
                )
            )
            if needs_initial_bank or needs_periodic_refresh:
                # ``eval_on_start=True`` normally initializes this bank. Keep
                # training self-contained when initial evaluation is disabled,
                # and periodically rebuild every row from the same model step.
                self._generate_item_embeddings(model)

            if (
                needs_privileged_item_context
                and getattr(self, "_opsd_frozen_teacher_model", None)
                is not None
                and getattr(
                    self,
                    "_opsd_frozen_teacher_item_hs",
                    None,
                )
                is None
            ):
                # Unlike the mutable student bank above, this snapshot is
                # generated exactly once from the frozen teacher adapter.
                self._generate_frozen_teacher_item_embeddings()

        student_batch = super()._generate_in_train(model, batch)
        if not self.opsd_enabled:
            return student_batch
        macro_batch_size = student_batch["multi_user_input_ids"].shape[0]
        self._make_mini_batch_divisible(macro_batch_size)

        num_samples = None
        if needs_item_context:
            original_batch_size = batch["seq_labels"].shape[0]
            if macro_batch_size % original_batch_size != 0:
                raise ValueError(
                    "Student rollouts cannot be aligned with recommendation labels."
                )
            num_samples = macro_batch_size // original_batch_size
            student_batch["opsd_seq_input_ids"] = batch[
                "seq_input_ids"
            ].repeat_interleave(num_samples, dim=0)
            student_batch["opsd_seq_labels"] = batch[
                "seq_labels"
            ].repeat_interleave(num_samples, dim=0)
            if self.lightgcn_enabled or self.small_moe_enabled:
                if "user_id" not in batch:
                    raise KeyError(
                        "LightGCN/MoE distillation requires user_id in the Recommendation batch."
                    )
                raw_user_ids = list(batch["user_id"])
                if len(raw_user_ids) != original_batch_size:
                    raise ValueError("user_id batch size does not match seq_labels.")
                student_batch["opsd_user_ids"] = [
                    user_id
                    for user_id in raw_user_ids
                    for _ in range(num_samples)
                ]

        hard_negative_ids = None
        if self.hard_negative_context_enabled:
            if num_samples is None:
                raise RuntimeError(
                    "HN-OPD could not align recommendation context."
                )
            hard_negative_ids = self._mine_online_student_hard_negatives(
                model=model,
                batch=batch,
                student_batch=student_batch,
                num_samples=num_samples,
            )

        if self._privileged_opsd_enabled:
            student_batch.update(self._build_teacher_batch(
                batch,
                student_batch,
                hard_negative_ids=hard_negative_ids,
            ))
        return student_batch

    @staticmethod
    def _rec_mini_batch_iterator(inputs, macro_batch_size, mini_batch_size):
        """Slice only fields consumed by the pure OPSD forward pass."""

        keys = [
            "multi_user_input_ids",
            "multi_user_attention_mask",
            "multi_user_labels",
            "opsd_teacher_input_ids",
            "opsd_teacher_attention_mask",
            "opsd_teacher_labels",
            "opsd_seq_input_ids",
            "opsd_seq_labels",
            "opsd_user_ids",
            "in_batch_labels",
        ]

        def _iterator():
            for start in range(0, macro_batch_size, mini_batch_size):
                end = start + mini_batch_size
                result = {
                    "seq_labels": inputs["seq_labels"],
                    "item_input_ids": inputs["item_input_ids"],
                    "item_attention_mask": inputs["item_attention_mask"],
                }
                for key in keys:
                    if key in inputs:
                        result[key] = inputs[key][start:end]
                yield result

        return _iterator()

    def _call_model_for_opsd(
            self,
            model,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            return_last_layer_attention: bool = False,
    ):
        """Run the transformer without materialising sequence-vocabulary logits."""

        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            return_causal_output=False,
            return_with_last_hidden_states=False,
            return_sequence_hidden_states=True,
            return_last_layer_attention=return_last_layer_attention,
        )
        if not return_last_layer_attention:
            return result
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(
                "teacher_last_layer_attention requires the Qwen OPSD model "
                "wrapper to return (sequence_hidden, final_attention)."
            )
        sequence_hidden, final_attention = result
        if not isinstance(sequence_hidden, torch.Tensor):
            raise RuntimeError("Qwen OPSD sequence hidden states are missing.")
        if not isinstance(final_attention, torch.Tensor):
            raise RuntimeError("Qwen final-layer attention is missing.")
        return sequence_hidden, final_attention

    def _get_output_head(self, model):
        """Resolve the shared LM head after Accelerate/DeepSpeed unwrapping."""

        unwrapped_model = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "get_base_model"):
            unwrapped_model = unwrapped_model.get_base_model()
        if not hasattr(unwrapped_model, "get_output_embeddings"):
            raise TypeError("The OPSD model does not expose get_output_embeddings().")
        output_head = unwrapped_model.get_output_embeddings()
        if output_head is None:
            raise TypeError("The OPSD model has no output embedding head.")
        return output_head

    def _record_teacher_topk_masses(self, masses: torch.Tensor):
        """Retain a bounded detached CPU sample for end-of-run diagnostics."""

        if not bool(getattr(
            self.args,
            "opsd_topk_mass_diagnostics",
            False,
        )):
            return
        values = masses.detach().float().reshape(-1)
        self._opsd_topk_mass_observed_positions += int(values.numel())
        remaining = (
            int(self.args.opsd_topk_mass_max_positions)
            - self._opsd_topk_mass_retained_positions
        )
        if remaining > 0 and values.numel() > 0:
            retained_values = values[:remaining].cpu().clone()
            self._opsd_topk_mass_sample_chunks.append(retained_values)
            self._opsd_topk_mass_retained_positions += int(
                retained_values.numel()
            )

    def _finalize_teacher_topk_mass_metrics(self):
        if self._opsd_topk_mass_sample_chunks:
            local_values = torch.cat(
                self._opsd_topk_mass_sample_chunks,
                dim=0,
            )
        else:
            local_values = torch.empty(0, dtype=torch.float32)
        payload = {
            "values": local_values.tolist(),
            "observed": int(self._opsd_topk_mass_observed_positions),
        }
        # This is a run-level diagnostic sample, not the current dataloader
        # batch. Avoid gather_for_metrics, whose end-of-dataloader remainder
        # trimming would incorecommendationtly drop whole rank payloads.
        gathered_payloads = gather_object([payload])
        gathered_values = []
        observed_positions = 0
        for gathered in gathered_payloads:
            gathered_values.extend(gathered["values"])
            observed_positions += int(gathered["observed"])

        metrics = _teacher_topk_mass_summary(torch.tensor(
            gathered_values,
            dtype=torch.float32,
        ))
        sampled_positions = int(
            metrics["opsd_teacher_topk_mass_sample_count"]
        )
        metrics["opsd_teacher_topk_mass_observed_count"] = float(
            observed_positions
        )
        metrics["opsd_teacher_topk_mass_sampling_fraction"] = (
            float(sampled_positions / observed_positions)
            if observed_positions > 0
            else 0.0
        )
        metrics["opsd_teacher_topk_mass_topk"] = float(self.args.opsd_topk)
        return metrics

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        if bool(getattr(
            self.args,
            "opsd_topk_mass_diagnostics",
            False,
        )):
            train_output.metrics.update(
                self._finalize_teacher_topk_mass_metrics()
            )
        return train_output

    def _compute_opsd_loss(self, model, batch, prefix: str):
        target_weight = float(self.args.opsd_target_weight)
        target_loss_type = _normalize_opsd_target_loss_type(
            getattr(self.args, "opsd_target_loss_type", "infonce")
        )
        token_weight = float(self.args.opsd_token_weight)
        token_credit_mode = _normalize_opsd_token_credit_mode(getattr(
            self.args,
            "opsd_token_credit_mode",
            "uniform",
        ))
        entropy_gate_mode = _normalize_opsd_entropy_gate_mode(getattr(
            self.args,
            "opsd_entropy_gate_mode",
            "none",
        ))
        teacher_gate_mode = _normalize_opsd_teacher_gate_mode(getattr(
            self.args,
            "opsd_teacher_gate_mode",
            "hard",
        ))
        rank_loss_type = _normalize_opsd_rank_loss_type(getattr(
            self.args,
            "opsd_rank_loss_type",
            "distribution_kl",
        ))
        rank_weight = self._effective_rank_weight
        lightgcn_weight = float(self.args.lightgcn_rank_weight)
        sasrec_weight = float(getattr(self.args, "sasrec_rank_weight", 0.0))
        small_moe_weight = float(getattr(
            self.args,
            "opsd_small_moe_weight",
            0.0,
        ))
        na_weight = float(getattr(self.args, "opsd_na_weight", 0.0))
        directau_weight = float(getattr(
            self.args,
            "opsd_directau_weight",
            0.0,
        ))
        rationale_geometry_weight = float(getattr(
            self.args,
            "opsd_rationale_geometry_weight",
            0.0,
        ))
        need_target_loss = target_weight != 0
        need_target_directau = (
            need_target_loss and target_loss_type == "privileged_directau"
        )
        need_independent_directau = directau_weight != 0
        need_rationale_geometry = rationale_geometry_weight != 0
        need_privileged_directau = (
            need_target_directau or need_independent_directau
        )
        need_na_loss = na_weight != 0
        need_token_loss = token_weight != 0
        need_entropy_gate = (
            need_token_loss and entropy_gate_mode != "none"
        )
        need_token_credit = (
            need_token_loss and token_credit_mode != "uniform"
        )
        need_target_item_credit = (
            need_token_loss and token_credit_mode == "target_item_cosine"
        )
        need_teacher_attention_credit = (
            need_token_loss
            and token_credit_mode == "teacher_last_layer_attention"
        )
        need_rank_loss = rank_weight != 0
        need_lightgcn_loss = lightgcn_weight != 0
        need_sasrec_loss = sasrec_weight != 0
        need_small_moe_loss = small_moe_weight != 0
        teacher_better_only = bool(self.args.opsd_teacher_better_only)
        need_advantage_weighted_gate = (
            teacher_better_only
            and teacher_gate_mode == "advantage_weighted"
            and (need_token_loss or need_rank_loss)
        )
        need_item_scores = need_rank_loss or (
            need_token_loss and teacher_better_only
        )
        need_privileged_teacher = (
            need_token_loss or need_rank_loss or need_privileged_directau
        )

        student_sequence_hidden = self._call_model_for_opsd(
            model,
            batch[f"{prefix}_input_ids"],
            batch[f"{prefix}_attention_mask"],
        )

        teacher_sequence_hidden = None
        teacher_last_layer_attention = None
        frozen_teacher_model = getattr(
            self, "_opsd_frozen_teacher_model", None
        )
        teacher_forward_model = (
            frozen_teacher_model
            if frozen_teacher_model is not None
            else model
        )
        if need_privileged_teacher:
            # ``no_grad`` alone does not disable dropout.  Normal Recommendation launches
            # already disable it, but this guard keeps the teacher deterministic.
            if teacher_forward_model is model:
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        teacher_result = self._call_model_for_opsd(
                            teacher_forward_model,
                            batch["opsd_teacher_input_ids"],
                            batch["opsd_teacher_attention_mask"],
                            return_last_layer_attention=(
                                need_teacher_attention_credit
                            ),
                        )
                        if need_teacher_attention_credit:
                            (
                                teacher_sequence_hidden,
                                teacher_last_layer_attention,
                            ) = teacher_result
                        else:
                            teacher_sequence_hidden = teacher_result
                finally:
                    model.train(was_training)
            else:
                # The snapshot is permanently in eval mode and owns separate
                # parameter storage, so optimizer steps on ``model`` cannot
                # change these teacher logits.
                with torch.no_grad():
                    teacher_result = self._call_model_for_opsd(
                        teacher_forward_model,
                        batch["opsd_teacher_input_ids"],
                        batch["opsd_teacher_attention_mask"],
                        return_last_layer_attention=(
                            need_teacher_attention_credit
                        ),
                    )
                    if need_teacher_attention_credit:
                        (
                            teacher_sequence_hidden,
                            teacher_last_layer_attention,
                        ) = teacher_result
                    else:
                        teacher_sequence_hidden = teacher_result

        zero_loss = student_sequence_hidden.sum() * 0.0
        target_loss = zero_loss
        item_loss = zero_loss
        lightgcn_loss = zero_loss
        sasrec_loss = zero_loss
        small_moe_loss = zero_loss
        recommendation_metrics = {}
        distill_mask = torch.ones(
            student_sequence_hidden.shape[0],
            dtype=torch.bool,
            device=student_sequence_hidden.device,
        )
        distill_weights = torch.ones(
            student_sequence_hidden.shape[0],
            dtype=torch.float32,
            device=student_sequence_hidden.device,
        )
        student_hidden = None
        teacher_hidden = None
        student_catalog_scores = None
        if (
            need_target_loss
            or need_na_loss
            or need_independent_directau
            or need_target_item_credit
            or need_item_scores
            or need_lightgcn_loss
            or need_sasrec_loss
            or need_small_moe_loss
            or need_rationale_geometry
        ):
            if self.item_hs is None:
                raise RuntimeError(
                    "Recommendation objectives require an initialized item bank."
                )
            student_hidden = _last_non_padding_hidden(
                student_sequence_hidden,
                batch[f"{prefix}_attention_mask"],
            )

        if need_target_loss or need_na_loss or need_independent_directau:
            if student_hidden is None:
                raise RuntimeError(
                    "Target recommendation and NA losses require student user "
                    "embeddings."
                )
            if "opsd_seq_labels" not in batch:
                raise KeyError(
                    "Target recommendation mini-batch is missing target item IDs."
                )
            if "seq_labels" not in batch:
                raise KeyError(
                    "Target recommendation batch is missing candidate item IDs."
                )

            live_student_item_hidden = None
            if need_privileged_directau:
                # Keep the local item forward alive for DirectAU. The cached
                # item bank is intentionally detached and therefore cannot
                # train the deployable student's item representation.
                live_student_item_hidden = model(
                    attention_mask=batch["item_attention_mask"],
                    input_ids=batch["item_input_ids"],
                    return_causal_output=False,
                    return_with_last_hidden_states=True,
                )

            in_batch_scores = None
            needs_in_batch_scores = (
                need_na_loss
                or (
                    need_target_loss
                    and target_loss_type in {"infonce", "full_softmax_ce"}
                )
            )
            if needs_in_batch_scores:
                # Reuse the original Recommendation dual-encoder path. Gradients flow
                # into the generated user and live item representations.
                in_batch_scores, _ = self.compute_sim_train(
                    model=model,
                    inputs=batch,
                    user_h=student_hidden,
                    item_h=live_student_item_hidden,
                    user_input_prefix=prefix,
                    update_item_bank=not self._use_consistent_item_bank,
                )
                item_device = in_batch_scores.device
            else:
                # DirectAU consumes the live positive item embeddings without
                # materialising an unused user-by-candidate score graph.
                item_device = live_student_item_hidden.device
                if not self._use_consistent_item_bank:
                    candidate_ids_for_bank = batch["seq_labels"].reshape(-1).to(
                        self.item_hs.device
                    )
                    self.item_hs[candidate_ids_for_bank] = (
                        live_student_item_hidden.detach().to(
                            device=self.item_hs.device,
                            dtype=self.item_hs.dtype,
                        )
                    )
            local_target_item_ids = batch["opsd_seq_labels"].reshape(-1).to(
                item_device
            )
            local_candidate_item_ids = batch["seq_labels"].reshape(-1).to(
                item_device
            )
            local_pair_group_ids = batch.get("in_batch_labels")
            if local_pair_group_ids is None:
                local_pair_group_ids = torch.arange(
                    local_target_item_ids.numel(),
                    device=item_device,
                )
            else:
                local_pair_group_ids = local_pair_group_ids.reshape(-1).to(
                    item_device
                )
            if local_pair_group_ids.shape[0] != local_target_item_ids.shape[0]:
                raise ValueError(
                    "LightCCF NA rollout group IDs do not match target IDs."
                )
            gathered_target_item_ids = local_target_item_ids
            gathered_candidate_item_ids = local_candidate_item_ids
            gathered_user_hidden = student_hidden
            gathered_pair_group_ids = local_pair_group_ids
            if (
                self.args.gather_negs_across_processes
                and self.accelerator.num_processes > 1
            ):
                # ``compute_sim_train`` uses the same local-first gather order.
                gathered_target_item_ids = self._gather_and_cat(
                    local_target_item_ids
                )
                gathered_candidate_item_ids = self._gather_and_cat(
                    local_candidate_item_ids
                )
                if need_na_loss:
                    gathered_user_hidden = self._gather_and_cat(student_hidden)
                    distributed_group_ids = (
                        local_pair_group_ids
                        + int(self.accelerator.process_index)
                        * int(local_candidate_item_ids.numel())
                    )
                    gathered_pair_group_ids = self._gather_and_cat(
                        distributed_group_ids
                    )

            privileged_directau_loss = None
            if need_privileged_directau:
                if live_student_item_hidden is None:
                    raise RuntimeError(
                        "Privileged DirectAU requires live student item "
                        "embeddings."
                    )
                if teacher_sequence_hidden is None:
                    raise RuntimeError(
                        "Privileged DirectAU requires teacher hidden states."
                    )
                teacher_hidden = _last_non_padding_hidden(
                    teacher_sequence_hidden,
                    batch["opsd_teacher_attention_mask"],
                )
                student_target_item_hidden = _select_target_item_embeddings(
                    live_student_item_hidden,
                    candidate_item_ids=local_candidate_item_ids,
                    target_item_ids=local_target_item_ids,
                )
                teacher_item_bank = self._get_teacher_item_bank_for_scoring(
                    self.item_hs.detach()
                )
                teacher_target_item_hidden = teacher_item_bank.index_select(
                    0,
                    local_target_item_ids.to(teacher_item_bank.device),
                )
                (
                    privileged_directau_loss,
                    directau_metrics,
                ) = _privileged_directau_loss(
                    student_user_hidden=student_hidden,
                    student_target_item_hidden=student_target_item_hidden,
                    teacher_user_hidden=teacher_hidden,
                    teacher_target_item_hidden=teacher_target_item_hidden,
                    target_item_ids=local_target_item_ids,
                    user_ids=batch.get("opsd_user_ids"),
                    alignment_mode=self.args.opsd_privileged_alignment_mode,
                    privileged_alignment_weight=float(
                        self.args.opsd_privileged_alignment_weight
                    ),
                    uniformity_weight=float(
                        self.args.opsd_directau_uniform_weight
                    ),
                    uniformity_temperature=float(
                        self.args.opsd_directau_uniform_temperature
                    ),
                )
                recommendation_metrics.update(directau_metrics)
                if need_independent_directau:
                    recommendation_metrics[
                        "_opsd_directau_loss_for_backward"
                    ] = privileged_directau_loss

            if need_target_loss:
                if target_loss_type == "infonce":
                    if in_batch_scores is None:
                        raise RuntimeError("Target InfoNCE scores were not built.")
                    target_loss, target_metrics = _target_item_infonce_loss(
                        in_batch_scores,
                        gathered_target_item_ids,
                        gathered_candidate_item_ids,
                    )
                elif target_loss_type == "full_softmax_ce":
                    if in_batch_scores is None:
                        raise RuntimeError(
                            "Full Softmax CE in-batch scores were not built."
                        )
                    if "opsd_seq_input_ids" not in batch:
                        raise KeyError(
                            "Full Softmax CE mini-batch is missing history item IDs."
                        )
                    local_user_count = int(student_hidden.shape[0])
                    local_candidate_count = int(local_candidate_item_ids.numel())
                    local_in_batch_scores = in_batch_scores[
                        :local_user_count,
                        :local_candidate_count,
                    ]
                    positive_matches = local_target_item_ids.unsqueeze(1).eq(
                        local_candidate_item_ids.unsqueeze(0)
                    )
                    if not positive_matches.any(dim=1).all():
                        raise ValueError(
                            "Every Full Softmax CE target must appear among the "
                            "current local item prompts."
                        )
                    live_target_columns = positive_matches.to(
                        dtype=torch.int64
                    ).argmax(dim=1)
                    live_target_scores = local_in_batch_scores.gather(
                        1,
                        live_target_columns.unsqueeze(1),
                    ).squeeze(1)
                    student_catalog_scores = self.similarity(
                        student_hidden,
                        self.item_hs.detach(),
                    )
                    target_loss, target_metrics = (
                        _target_item_full_softmax_ce_loss(
                            student_scores=student_catalog_scores,
                            target_item_ids=local_target_item_ids,
                            history_item_ids=batch["opsd_seq_input_ids"],
                            live_target_scores=live_target_scores,
                            padding_item_id=0,
                        )
                    )
                else:
                    if privileged_directau_loss is None:
                        raise RuntimeError(
                            "Privileged DirectAU target loss was not built."
                        )
                    target_loss = privileged_directau_loss
                    target_metrics = {}
                recommendation_metrics.update(target_metrics)

            if need_na_loss:
                na_loss, na_metrics = (
                    _lightccf_neighborhood_aggregation_loss(
                        in_batch_item_scores=in_batch_scores,
                        paired_user_hidden=gathered_user_hidden,
                        paired_target_item_ids=gathered_target_item_ids,
                        candidate_item_ids=gathered_candidate_item_ids,
                        similarity_temperature=float(self.similarity.temp),
                        na_temperature=float(
                            getattr(self.args, "opsd_na_temperature", 0.2)
                        ),
                        paired_group_ids=gathered_pair_group_ids,
                    )
                )
                recommendation_metrics.update(na_metrics)
                recommendation_metrics["_opsd_na_loss_for_backward"] = na_loss

        if need_item_scores:
            if teacher_sequence_hidden is None:
                raise RuntimeError("Privileged item scoring requires teacher hidden states.")
            if teacher_hidden is None:
                teacher_hidden = _last_non_padding_hidden(
                    teacher_sequence_hidden,
                    batch["opsd_teacher_attention_mask"],
                )

            if (
                not (need_target_loss or need_na_loss)
                and not self._use_consistent_item_bank
            ):
                # The target/NA path above already refreshed these rows with current item
                # embeddings.  Without it, retain the old gradient-free refresh
                # used by legacy pure privileged-ranking distillation. A
                # consistent OPSD bank is rebuilt only as one full snapshot.
                with torch.no_grad():
                    self.compute_sim_train(
                        model=model,
                        inputs=batch,
                        user_h=student_hidden.detach(),
                        user_input_prefix=prefix,
                    )

            item_bank = self.item_hs.detach()
            if need_rank_loss and rank_loss_type == "distribution_kl":
                student_scores = self.similarity(student_hidden, item_bank)
            else:
                with torch.no_grad():
                    student_scores = self.similarity(
                        student_hidden.detach(), item_bank
                    )
            teacher_item_bank = self._get_teacher_item_bank_for_scoring(
                item_bank
            )
            with torch.no_grad():
                teacher_scores = self.similarity(
                    teacher_hidden,
                    teacher_item_bank,
                )
            reward_metrics, distill_mask, reward_per_example = (
                _recommendation_reward_diagnostics(
                    student_scores=student_scores,
                    teacher_scores=teacher_scores,
                    history_item_ids=batch["opsd_seq_input_ids"],
                    target_item_ids=batch["opsd_seq_labels"],
                    reward_ndcg_k=self.args.reward_ndcg_k,
                    teacher_better_only=teacher_better_only,
                    return_per_example=True,
                )
            )
            recommendation_metrics.update(reward_metrics)
            distill_weights, gate_metrics = (
                _recommendation_advantage_gate_weights(
                    reward_gap=reward_per_example["teacher_gate_signal"],
                    enabled=teacher_better_only,
                    mode=teacher_gate_mode,
                    temperature=float(
                        self.args.opsd_teacher_gate_temperature
                    ),
                    power=float(self.args.opsd_teacher_gate_power),
                )
            )
            recommendation_metrics.update(gate_metrics)
            recommendation_metrics["opsd_teacher_item_bank_isolated"] = (
                student_scores.new_tensor(
                    float(frozen_teacher_model is not None)
                )
            )

            if need_rank_loss:
                if rank_loss_type == "distribution_kl":
                    item_loss, item_kd_metrics = (
                        _privileged_item_rank_distillation_loss(
                            student_scores=student_scores,
                            teacher_scores=teacher_scores,
                            history_item_ids=batch["opsd_seq_input_ids"],
                            target_item_ids=batch["opsd_seq_labels"],
                            distill_mask=distill_mask,
                            distill_weights=(
                                distill_weights
                                if need_advantage_weighted_gate
                                else None
                            ),
                            topk=int(self.args.opsd_rank_topk),
                            temperature=float(self.args.opsd_rank_temperature),
                        )
                    )
                else:
                    item_loss, item_kd_metrics = (
                        _privileged_item_geometry_distillation_loss(
                            student_user_hidden=student_hidden,
                            teacher_user_hidden=teacher_hidden,
                            student_item_hs=item_bank,
                            teacher_item_hs=teacher_item_bank,
                            student_scores=student_scores,
                            teacher_scores=teacher_scores,
                            history_item_ids=batch["opsd_seq_input_ids"],
                            target_item_ids=batch["opsd_seq_labels"],
                            distill_mask=distill_mask,
                            distill_weights=(
                                distill_weights
                                if need_advantage_weighted_gate
                                else None
                            ),
                            topk=int(self.args.opsd_rank_topk),
                            positive_margin_only=bool(getattr(
                                self.args,
                                "opsd_geometry_positive_margin_only",
                                True,
                            )),
                        )
                    )
                recommendation_metrics.update(item_kd_metrics)

        if need_lightgcn_loss:
            if self.lightgcn_bank is None or student_hidden is None:
                raise RuntimeError("LightGCN embedding bank was not initialized.")
            if "opsd_user_ids" not in batch:
                raise KeyError("LightGCN mini-batch is missing opsd_user_ids.")
            item_bank = self.item_hs.detach()
            # Undo the project's base similarity temperature so the dedicated
            # student temperature below has an intuitive raw-score meaning.
            if student_catalog_scores is None:
                student_catalog_scores = self.similarity(
                    student_hidden,
                    item_bank,
                )
            student_cf_scores = (
                student_catalog_scores * float(self.similarity.temp)
            )
            teacher_cf_scores, mapped_user_mask = (
                self.lightgcn_bank.score_users(batch["opsd_user_ids"])
            )
            lightgcn_loss, lightgcn_metrics = (
                lightgcn_rank_distillation_loss(
                    student_scores=student_cf_scores,
                    teacher_scores=teacher_cf_scores,
                    history_item_ids=batch["opsd_seq_input_ids"],
                    target_item_ids=batch["opsd_seq_labels"],
                    valid_user_mask=mapped_user_mask,
                    valid_item_mask=self.lightgcn_bank.train_item_mask,
                    topk=int(self.args.lightgcn_topk),
                    teacher_temperature=float(
                        self.args.lightgcn_teacher_temperature
                    ),
                    student_temperature=float(
                        self.args.lightgcn_student_temperature
                    ),
                    teacher_better_only=bool(
                        self.args.lightgcn_teacher_better_only
                    ),
                )
            )
            recommendation_metrics.update(lightgcn_metrics)

        if need_sasrec_loss:
            if self.sasrec_teacher is None or student_hidden is None:
                raise RuntimeError("SASRec teacher was not initialized.")
            if "opsd_seq_input_ids" not in batch:
                raise KeyError("SASRec mini-batch is missing history item IDs.")
            item_bank = self.item_hs.detach()
            # Match the LightGCN branch: first recover raw Recommendation dot scores,
            # then apply the dedicated student temperature in the KL helper.
            if student_catalog_scores is None:
                student_catalog_scores = self.similarity(
                    student_hidden,
                    item_bank,
                )
            student_sequential_scores = (
                student_catalog_scores * float(self.similarity.temp)
            )
            teacher_sequential_scores = self.sasrec_teacher.score_histories(
                batch["opsd_seq_input_ids"]
            )
            sasrec_loss, sasrec_metrics = sasrec_rank_distillation_loss(
                student_scores=student_sequential_scores,
                teacher_scores=teacher_sequential_scores,
                history_item_ids=batch["opsd_seq_input_ids"],
                target_item_ids=batch["opsd_seq_labels"],
                valid_item_mask=self.sasrec_teacher.train_item_mask,
                topk=int(getattr(self.args, "sasrec_topk", 32)),
                teacher_temperature=float(
                    getattr(self.args, "sasrec_teacher_temperature", 0.2)
                ),
                student_temperature=float(
                    getattr(self.args, "sasrec_student_temperature", 0.2)
                ),
                teacher_better_only=bool(
                    getattr(self.args, "sasrec_teacher_better_only", True)
                ),
                teacher_gate_mode=str(getattr(
                    self.args,
                    "sasrec_teacher_gate_mode",
                    "hard",
                )),
                teacher_reward_mode=str(getattr(
                    self.args,
                    "sasrec_teacher_reward_mode",
                    "normalized_log_rank",
                )),
                teacher_gate_temperature=float(getattr(
                    self.args,
                    "sasrec_teacher_gate_temperature",
                    0.1,
                )),
                teacher_gate_power=float(getattr(
                    self.args,
                    "sasrec_teacher_gate_power",
                    1.0,
                )),
            )
            recommendation_metrics.update(sasrec_metrics)
            # Preserve this private method's established five-value return.
            # batch_forward consumes and removes the live scalar immediately.
            recommendation_metrics["_sasrec_rank_loss_for_backward"] = (
                sasrec_loss
            )

        if need_small_moe_loss:
            if self.small_teacher_moe is None or student_hidden is None:
                raise RuntimeError(
                    "The independent four-expert MoE was not initialized."
                )
            if "opsd_seq_input_ids" not in batch:
                raise KeyError("Small-teacher MoE is missing history item IDs.")
            if "opsd_user_ids" not in batch:
                raise KeyError("Small-teacher MoE is missing user IDs.")
            item_bank = self.item_hs.detach()
            if student_catalog_scores is None:
                student_catalog_scores = self.similarity(
                    student_hidden,
                    item_bank,
                )
            # Undo the base similarity temperature. The MoE calibrates the
            # student and all four experts independently below.
            student_moe_scores = (
                student_catalog_scores * float(self.similarity.temp)
            )
            teacher_outputs = self.small_teacher_moe.score(
                history_item_ids=batch["opsd_seq_input_ids"],
                user_ids=batch["opsd_user_ids"],
                student_device=student_moe_scores.device,
            )
            mechanism_capture = getattr(
                self,
                "_mechanism_analysis_capture",
                None,
            )
            small_moe_result = small_teacher_moe_distillation_loss(
                    student_scores=student_moe_scores,
                    teacher_outputs=teacher_outputs,
                    history_item_ids=batch["opsd_seq_input_ids"],
                    target_item_ids=batch["opsd_seq_labels"],
                    topk=int(self.args.opsd_small_moe_topk),
                    student_temperature=float(
                        self.args.opsd_small_moe_student_temperature
                    ),
                    teacher_temperatures={
                        "sasrec": float(
                            self.args.opsd_small_moe_sasrec_temperature
                        ),
                        "lightgcn": float(
                            self.args.opsd_small_moe_lightgcn_temperature
                        ),
                        "fmlprec": float(
                            self.args.opsd_small_moe_fmlprec_temperature
                        ),
                        "ease": float(
                            self.args.opsd_small_moe_ease_temperature
                        ),
                    },
                    gate_temperature=float(
                        self.args.opsd_small_moe_gate_temperature
                    ),
                    outer_temperature=float(
                        self.args.opsd_small_moe_outer_temperature
                    ),
                    quality_power=float(
                        self.args.opsd_small_moe_quality_power
                    ),
                    router_mode=self.args.opsd_small_moe_router_mode,
                    advantage_mode=self.args.opsd_small_moe_advantage_mode,
                    return_per_example=callable(mechanism_capture),
                )
            if callable(mechanism_capture):
                (
                    small_moe_loss,
                    small_moe_metrics,
                    small_moe_per_example,
                ) = small_moe_result
                mechanism_capture("expert", small_moe_per_example)
            else:
                small_moe_loss, small_moe_metrics = small_moe_result
            recommendation_metrics.update(small_moe_metrics)
            # Keep the established five-value return of this private method.
            recommendation_metrics["_opsd_small_moe_loss_for_backward"] = (
                small_moe_loss
            )

        if need_rationale_geometry:
            if self.item_hs is None:
                raise RuntimeError(
                    "Rationale geometry requires an initialized item bank."
                )
            if "opsd_seq_labels" not in batch:
                raise KeyError(
                    "Rationale geometry is missing target item IDs."
                )
            marker = str(getattr(self.args, "emb_token", ""))
            marker_token_ids = self.processing_class.encode(
                marker,
                add_special_tokens=False,
            )
            if not marker_token_ids:
                raise RuntimeError(
                    "Rationale geometry requires a non-empty encoded "
                    "emb_token marker."
                )
            rationale_hidden, valid_rationales, rationale_counts = (
                _mean_reasoning_token_hidden(
                    sequence_hidden=student_sequence_hidden,
                    labels=batch[f"{prefix}_labels"],
                    marker_token_count=len(marker_token_ids),
                )
            )
            target_item_ids = batch["opsd_seq_labels"].reshape(-1).to(
                device=self.item_hs.device,
                dtype=torch.long,
            )
            if target_item_ids.shape[0] != rationale_hidden.shape[0]:
                raise ValueError(
                    "Rationale and target-item batch sizes differ."
                )
            if (
                (target_item_ids <= 0).any()
                or (target_item_ids >= self.item_hs.shape[0]).any()
            ):
                raise ValueError(
                    "Rationale geometry received a padding or out-of-range "
                    "target item ID."
                )
            target_item_hidden = self.item_hs.index_select(
                0,
                target_item_ids,
            ).detach()
            valid_on_rationale_device = valid_rationales.to(
                rationale_hidden.device
            )
            valid_on_item_device = valid_rationales.to(
                target_item_hidden.device
            )
            valid_target_ids = target_item_ids[
                valid_rationales.to(target_item_ids.device)
            ]
            rationale_geometry_loss, rationale_geometry_metrics = (
                _rationale_item_geometry_loss(
                    rationale_hidden=rationale_hidden[
                        valid_on_rationale_device
                    ],
                    target_item_hidden=target_item_hidden[
                        valid_on_item_device
                    ],
                    target_item_ids=valid_target_ids,
                    reason_temperature=float(
                        self.args.opsd_rationale_reason_temperature
                    ),
                    item_temperature=float(
                        self.args.opsd_rationale_item_temperature
                    ),
                    loss_type=str(
                        self.args.opsd_rationale_geometry_loss_type
                    ),
                )
            )
            metric_zero = rationale_counts[:0].sum().float()
            valid_counts = rationale_counts[valid_rationales]
            recommendation_metrics.update(rationale_geometry_metrics)
            recommendation_metrics.update({
                "opsd_rationale_geometry_valid_rate": (
                    valid_rationales.float().mean()
                ),
                "opsd_rationale_geometry_reasoning_token_count": (
                    valid_counts.float().mean()
                    if valid_counts.numel() > 0
                    else metric_zero
                ),
            })
            recommendation_metrics[
                "_opsd_rationale_geometry_loss_for_backward"
            ] = rationale_geometry_loss

        token_loss = zero_loss
        if need_token_loss:
            if teacher_sequence_hidden is None:
                raise RuntimeError("Token OPSD requires teacher hidden states.")
            token_distill_mask = distill_mask
            if need_advantage_weighted_gate:
                # Keep all completion positions so zero/small continuous
                # weights affect the absolute OPD strength rather than being
                # removed and renormalized as a hard subset.
                token_distill_mask = torch.ones_like(distill_mask)
            student_labels = batch[f"{prefix}_labels"].masked_fill(
                ~token_distill_mask.unsqueeze(1), -100
            )
            teacher_labels = batch["opsd_teacher_labels"].masked_fill(
                ~token_distill_mask.unsqueeze(1), -100
            )
            student_completion_hidden, student_targets, student_counts = (
                _select_completion_prediction_hidden(
                    student_sequence_hidden,
                    student_labels,
                )
            )
            teacher_completion_hidden, teacher_targets, teacher_counts = (
                _select_completion_prediction_hidden(
                    teacher_sequence_hidden,
                    teacher_labels,
                )
            )
            if not torch.equal(student_counts, teacher_counts):
                raise ValueError(
                    "Student and teacher completion lengths differ after batching."
                )
            if not torch.equal(student_targets, teacher_targets):
                raise ValueError(
                    "Student and teacher completion token ids are not identical."
                )

            max_token_kl = getattr(self.args, "opsd_max_token_kl", None)
            topk_mass_diagnostics = bool(getattr(
                self.args,
                "opsd_topk_mass_diagnostics",
                False,
            ))
            need_teacher_topk_mass = (
                topk_mass_diagnostics or need_entropy_gate
            )
            token_loss_or_values = _completion_token_distillation_loss(
                student_hidden=student_completion_hidden,
                teacher_hidden=teacher_completion_hidden,
                target_ids=student_targets,
                student_output_head=self._get_output_head(model),
                teacher_output_head=self._get_output_head(
                    teacher_forward_model
                ),
                temperature=self.args.opsd_temperature,
                loss_type=self.args.opsd_loss_type,
                topk=self.args.opsd_topk,
                chunk_size=self.args.opsd_token_chunk_size,
                return_per_token=(
                    max_token_kl is not None
                    or need_token_credit
                    or need_advantage_weighted_gate
                    or need_entropy_gate
                ),
                return_teacher_topk_mass=need_teacher_topk_mass,
                return_token_entropies=need_entropy_gate,
            )
            teacher_topk_masses = None
            student_token_entropies = None
            teacher_token_entropies = None
            if need_entropy_gate:
                (
                    token_loss_or_values,
                    teacher_topk_masses,
                    student_token_entropies,
                    teacher_token_entropies,
                ) = token_loss_or_values
            elif topk_mass_diagnostics:
                token_loss_or_values, teacher_topk_masses = (
                    token_loss_or_values
                )
            if topk_mass_diagnostics:
                marker_token_ids = self.processing_class.encode(
                    str(getattr(self.args, "emb_token", "")),
                    add_special_tokens=False,
                )
                if not marker_token_ids:
                    raise RuntimeError(
                        "Teacher Top-K mass diagnostics require a non-empty "
                        "encoded emb_token marker."
                    )
                reasoning_position_mask = (
                    _reasoning_position_mask_from_counts(
                        per_example_counts=student_counts,
                        marker_token_count=len(marker_token_ids),
                    )
                )
                if reasoning_position_mask.numel() != teacher_topk_masses.numel():
                    raise RuntimeError(
                        "Teacher Top-K masses do not align with completion "
                        "positions."
                    )
                self._record_teacher_topk_masses(
                    teacher_topk_masses[
                        reasoning_position_mask.to(
                            teacher_topk_masses.device
                        )
                    ]
                )
            if (
                not need_token_credit
                and not need_entropy_gate
                and not need_advantage_weighted_gate
                and max_token_kl is None
            ):
                # Keep the original reduction path bit-for-bit when disabled.
                token_loss = token_loss_or_values
            elif (
                not need_token_credit
                and not need_advantage_weighted_gate
                and not need_entropy_gate
            ):
                token_loss, token_filter_metrics = (
                    _filter_token_kl_by_example(
                        per_token_kl=token_loss_or_values,
                        per_example_counts=student_counts,
                        max_token_kl=float(max_token_kl),
                    )
                )
                recommendation_metrics.update(token_filter_metrics)
            else:
                selected_token_kl = token_loss_or_values
                selected_token_hidden = student_completion_hidden
                selected_token_targets = student_targets
                selected_counts = student_counts
                selected_student_entropy = student_token_entropies
                selected_teacher_entropy = teacher_token_entropies
                selected_teacher_topk_mass = teacher_topk_masses
                selected_examples = torch.ones(
                    student_counts.shape[0],
                    dtype=torch.bool,
                    device=student_counts.device,
                )
                if max_token_kl is not None:
                    (
                        _,
                        token_filter_metrics,
                        kept_tokens,
                        selected_examples,
                    ) = _filter_token_kl_by_example(
                        per_token_kl=token_loss_or_values,
                        per_example_counts=student_counts,
                        max_token_kl=float(max_token_kl),
                        return_selection=True,
                    )
                    recommendation_metrics.update(token_filter_metrics)
                    selected_token_kl = token_loss_or_values[kept_tokens]
                    selected_token_hidden = student_completion_hidden[
                        kept_tokens
                    ]
                    selected_token_targets = student_targets[kept_tokens]
                    selected_counts = student_counts[selected_examples]
                    if need_entropy_gate:
                        selected_student_entropy = student_token_entropies[
                            kept_tokens
                        ]
                        selected_teacher_entropy = teacher_token_entropies[
                            kept_tokens
                        ]
                        selected_teacher_topk_mass = teacher_topk_masses[
                            kept_tokens
                        ]

                uniform_reference = (
                    selected_token_kl.mean()
                    if selected_token_kl.numel() > 0
                    else selected_token_kl[:0].sum()
                )
                weighted_token_kl = selected_token_kl
                if need_target_item_credit:
                    if self.item_hs is None:
                        raise RuntimeError(
                            "Recommendation-aware token credit requires an "
                            "initialized item bank."
                        )
                    if "opsd_seq_labels" not in batch:
                        raise KeyError(
                            "Recommendation-aware token credit is missing "
                            "target item IDs."
                        )
                    target_item_ids = batch["opsd_seq_labels"].reshape(-1).to(
                        device=self.item_hs.device,
                        dtype=torch.long,
                    )
                    if target_item_ids.shape[0] != student_counts.shape[0]:
                        raise ValueError(
                            "Target-item batch size differs from token-count "
                            "batch."
                        )
                    if (
                        (target_item_ids < 0).any()
                        or (target_item_ids >= self.item_hs.shape[0]).any()
                    ):
                        raise ValueError(
                            "Recommendation-aware token credit received an "
                            "out-of-range target item ID."
                        )
                    target_item_hidden = self.item_hs.index_select(
                        0, target_item_ids
                    ).detach()
                    target_item_hidden = target_item_hidden[
                        selected_examples.to(target_item_hidden.device)
                    ]
                    token_credit_weights, token_credit_metrics = (
                        _target_item_token_credit_weights(
                            token_hidden=selected_token_hidden,
                            target_item_hidden=target_item_hidden,
                            per_example_counts=selected_counts,
                            temperature=(
                                self.args.opsd_token_credit_temperature
                            ),
                            mix=self.args.opsd_token_credit_mix,
                        )
                    )
                    weighted_token_kl = (
                        token_credit_weights.to(selected_token_kl.device)
                        * selected_token_kl
                    )
                    recommendation_metrics.update(token_credit_metrics)
                elif need_teacher_attention_credit:
                    if teacher_last_layer_attention is None:
                        raise RuntimeError(
                            "Teacher attention credit requires captured final-"
                            "layer attention."
                        )
                    marker = str(getattr(self.args, "emb_token", ""))
                    marker_token_ids = self.processing_class.encode(
                        marker,
                        add_special_tokens=False,
                    )
                    if not marker_token_ids:
                        raise RuntimeError(
                            "Teacher attention credit requires a non-empty "
                            "encoded emb_token marker."
                        )
                    selected_example_indices = selected_examples.to(
                        device=teacher_last_layer_attention.device,
                    )
                    selected_teacher_attention = teacher_last_layer_attention[
                        selected_example_indices
                    ]
                    selected_teacher_labels = teacher_labels[
                        selected_examples.to(teacher_labels.device)
                    ]
                    selected_teacher_attention_mask = batch[
                        "opsd_teacher_attention_mask"
                    ][selected_examples.to(
                        batch["opsd_teacher_attention_mask"].device
                    )]
                    token_credit_weights, token_credit_metrics = (
                        _teacher_attention_token_credit_weights(
                            last_layer_attention=(
                                selected_teacher_attention
                            ),
                            labels=selected_teacher_labels,
                            attention_mask=(
                                selected_teacher_attention_mask
                            ),
                            marker_token_count=len(marker_token_ids),
                            mix=self.args.opsd_token_credit_mix,
                        )
                    )
                    if token_credit_weights.numel() != selected_token_kl.numel():
                        raise RuntimeError(
                            "Teacher attention credits do not align with the "
                            "selected per-token KL values."
                        )
                    weighted_token_kl = (
                        token_credit_weights.to(selected_token_kl.device)
                        * selected_token_kl
                    )
                    recommendation_metrics.update(token_credit_metrics)

                entropy_gate_weights = None
                if need_entropy_gate:
                    entropy_gate_weights, entropy_gate_metrics = (
                        _entropy_gap_token_weights(
                            student_entropy=selected_student_entropy,
                            teacher_entropy=selected_teacher_entropy,
                            teacher_topk_mass=selected_teacher_topk_mass,
                            gap_threshold=(
                                self.args.opsd_entropy_gap_threshold
                            ),
                            min_teacher_topk_mass=(
                                self.args.opsd_entropy_gate_min_teacher_topk_mass
                            ),
                            mode=self.args.opsd_entropy_gate_mode,
                            temperature=(
                                self.args.opsd_entropy_gate_temperature
                            ),
                        )
                    )
                    recommendation_metrics.update(entropy_gate_metrics)
                    mechanism_capture = getattr(
                        self,
                        "_mechanism_analysis_capture",
                        None,
                    )
                    if callable(mechanism_capture):
                        mechanism_capture("entropy", {
                            "student_entropy": (
                                selected_student_entropy.detach()
                            ),
                            "teacher_entropy": (
                                selected_teacher_entropy.detach()
                            ),
                            "teacher_topk_mass": (
                                selected_teacher_topk_mass.detach()
                            ),
                            "gate_weight": entropy_gate_weights.detach(),
                            "token_ids": selected_token_targets.detach(),
                            "per_example_counts": selected_counts.detach(),
                            "selected_examples": selected_examples.detach(),
                        })

                if need_advantage_weighted_gate:
                    selected_gate_weights = distill_weights[
                        selected_examples.to(distill_weights.device)
                    ]
                    if entropy_gate_weights is not None:
                        candidate_counts = selected_counts.to(
                            device=weighted_token_kl.device,
                            dtype=torch.long,
                        )
                        candidate_token_weights = torch.repeat_interleave(
                            selected_gate_weights.detach().to(
                            device=weighted_token_kl.device,
                                dtype=weighted_token_kl.dtype,
                            ),
                            candidate_counts,
                        )
                        effective_token_weights = (
                            candidate_token_weights
                            * entropy_gate_weights.to(
                                device=weighted_token_kl.device,
                                dtype=weighted_token_kl.dtype
                            )
                        )
                        token_loss = _reduce_entropy_weighted_token_loss(
                            per_token_loss=weighted_token_kl,
                            token_weights=effective_token_weights,
                            normalization=getattr(
                                self.args,
                                "opsd_entropy_gate_loss_normalization",
                            ),
                        )
                        effective_token_weight = (
                            effective_token_weights.mean()
                            if effective_token_weights.numel() > 0
                            else weighted_token_kl[:0].sum()
                        )
                        gate_token_metrics = {
                            "opsd_teacher_gate_effective_token_weight": (
                                effective_token_weight.detach()
                            ),
                        }
                    else:
                        token_loss, gate_token_metrics = (
                            _weight_completion_token_loss_by_example(
                                per_token_loss=weighted_token_kl,
                                per_example_counts=selected_counts,
                                per_example_weights=selected_gate_weights,
                            )
                        )
                    recommendation_metrics.update(gate_token_metrics)
                elif entropy_gate_weights is not None:
                    token_loss = _reduce_entropy_weighted_token_loss(
                        per_token_loss=weighted_token_kl,
                        token_weights=entropy_gate_weights,
                        normalization=(
                            self.args.opsd_entropy_gate_loss_normalization
                        ),
                    )
                elif weighted_token_kl.numel() > 0:
                    token_loss = weighted_token_kl.mean()
                else:
                    token_loss = weighted_token_kl[:0].sum()
                recommendation_metrics[
                    "opsd_token_loss_uniform_reference"
                ] = uniform_reference.detach()
                if entropy_gate_weights is not None:
                    recommendation_metrics[
                        "opsd_entropy_gate_loss_scale"
                    ] = entropy_gate_weights.float().mean().detach()

        return (
            target_loss,
            token_loss,
            item_loss,
            lightgcn_loss,
            recommendation_metrics,
        )

    def batch_forward(self, model, batch, prefix):
        """Optimize the selected target loss and distillation objectives."""
        if not self.opsd_enabled:
            return torch.zeros(
                (), device=self.args.device, requires_grad=True)

        (
            target_loss,
            token_loss,
            item_loss,
            lightgcn_loss,
            recommendation_metrics,
        ) = (
            self._compute_opsd_loss(model, batch, prefix)
        )
        target_weight = float(self.args.opsd_target_weight)
        rank_weight = self._effective_rank_weight
        lightgcn_weight = float(self.args.lightgcn_rank_weight)
        sasrec_weight = float(getattr(self.args, "sasrec_rank_weight", 0.0))
        small_moe_weight = float(getattr(
            self.args,
            "opsd_small_moe_weight",
            0.0,
        ))
        na_weight = float(getattr(self.args, "opsd_na_weight", 0.0))
        directau_weight = float(getattr(
            self.args,
            "opsd_directau_weight",
            0.0,
        ))
        rationale_geometry_weight = float(getattr(
            self.args,
            "opsd_rationale_geometry_weight",
            0.0,
        ))
        sasrec_loss = recommendation_metrics.pop(
            "_sasrec_rank_loss_for_backward",
            lightgcn_loss * 0.0,
        )
        small_moe_loss = recommendation_metrics.pop(
            "_opsd_small_moe_loss_for_backward",
            lightgcn_loss * 0.0,
        )
        na_loss = recommendation_metrics.pop(
            "_opsd_na_loss_for_backward",
            target_loss * 0.0,
        )
        directau_loss = recommendation_metrics.pop(
            "_opsd_directau_loss_for_backward",
            target_loss * 0.0,
        )
        rationale_geometry_loss = recommendation_metrics.pop(
            "_opsd_rationale_geometry_loss_for_backward",
            target_loss * 0.0,
        )
        directau_loss_for_backward = directau_loss
        target_gradient_scale = 1.0
        target_loss_type = _normalize_opsd_target_loss_type(
            getattr(self.args, "opsd_target_loss_type", "infonce")
        )
        if (
            target_loss_type == "infonce"
            and
            target_weight != 0
            and self.args.gather_negs_across_processes
            and self.accelerator.num_processes > 1
        ):
            # Each rank owns gradients only for its local gathered embeddings,
            # while the target loss is already averaged over all ranks' rows.
            # Distributed training averages those partial parameter gradients
            # once more, so restore their intended global-objective magnitude.
            target_gradient_scale = float(self.accelerator.num_processes)
        target_loss_for_backward = _scale_loss_gradient(
            target_loss,
            target_gradient_scale,
        )
        na_gradient_scale = 1.0
        if (
            na_weight != 0
            and self.args.gather_negs_across_processes
            and self.accelerator.num_processes > 1
        ):
            na_gradient_scale = float(self.accelerator.num_processes)
        na_loss_for_backward = _scale_loss_gradient(
            na_loss,
            na_gradient_scale,
        )
        text_space_loss = self.args.opsd_token_weight * token_loss
        gradient_coordination_mode = (
            _normalize_opsd_gradient_coordination_mode(getattr(
                self.args,
                "opsd_gradient_coordination_mode",
                "sum",
            ))
        )
        if gradient_coordination_mode == "sum":
            # Preserve the original path exactly: no parameter enumeration,
            # autograd probes, distributed collectives, or reassociation of
            # the established floating-point additions.
            total_loss = (
                target_weight * target_loss_for_backward
                + na_weight * na_loss_for_backward
                + directau_weight * directau_loss_for_backward
                + self.args.opsd_token_weight * token_loss
                + rank_weight * item_loss
                + lightgcn_weight * lightgcn_loss
                + sasrec_weight * sasrec_loss
                + small_moe_weight * small_moe_loss
            )
            recommendation_space_loss = (
                target_weight * target_loss.detach()
                + na_weight * na_loss.detach()
                + directau_weight * directau_loss.detach()
                + rank_weight * item_loss.detach()
                + lightgcn_weight * lightgcn_loss.detach()
                + sasrec_weight * sasrec_loss.detach()
                + small_moe_weight * small_moe_loss.detach()
            )
            if rationale_geometry_weight != 0:
                total_loss = total_loss + (
                    rationale_geometry_weight * rationale_geometry_loss
                )
                recommendation_space_loss = recommendation_space_loss + (
                    rationale_geometry_weight
                    * rationale_geometry_loss.detach()
                )
        else:
            recommendation_space_loss = (
                target_weight * target_loss_for_backward
                + na_weight * na_loss_for_backward
                + directau_weight * directau_loss_for_backward
                + rank_weight * item_loss
                + lightgcn_weight * lightgcn_loss
                + sasrec_weight * sasrec_loss
                + small_moe_weight * small_moe_loss
            )
            if rationale_geometry_weight != 0:
                recommendation_space_loss = recommendation_space_loss + (
                    rationale_geometry_weight * rationale_geometry_loss
                )
            gradient_reducer = None
            if self.accelerator.num_processes > 1:
                def gradient_reducer(flat_gradient):
                    return self.accelerator.reduce(
                        flat_gradient,
                        reduction="mean",
                    )

            total_loss, gradient_metrics = (
                _recommendation_preserving_gradient_surrogate(
                    recommendation_loss=recommendation_space_loss,
                    text_loss=text_space_loss,
                    parameters=model.parameters(),
                    mode=gradient_coordination_mode,
                    epsilon=self.args.opsd_gradient_coordination_epsilon,
                    gradient_reducer=gradient_reducer,
                )
            )
            recommendation_metrics.update(gradient_metrics)

        metrics = {
            "opsd_target_recommendation_loss": (
                target_loss.detach().float().item()
            ),
            "opsd_target_infonce_loss": target_loss.detach().float().item(),
            "opsd_na_loss": na_loss.detach().float().item(),
            "opsd_directau_loss": directau_loss.detach().float().item(),
            "opsd_token_loss": token_loss.detach().float().item(),
            "opsd_item_loss": item_loss.detach().float().item(),
            "lightgcn_rank_loss": lightgcn_loss.detach().float().item(),
            "sasrec_rank_loss": sasrec_loss.detach().float().item(),
            "opsd_small_moe_loss": small_moe_loss.detach().float().item(),
            "opsd_weighted_target_infonce_loss": (
                target_weight * target_loss
            ).detach().float().item(),
            "opsd_weighted_target_recommendation_loss": (
                target_weight * target_loss
            ).detach().float().item(),
            "opsd_target_infonce_gradient_scale": target_gradient_scale,
            "opsd_target_recommendation_gradient_scale": (
                target_gradient_scale
            ),
            "opsd_na_gradient_scale": na_gradient_scale,
            "opsd_weighted_na_loss": (
                na_weight * na_loss
            ).detach().float().item(),
            "opsd_weighted_directau_loss": (
                directau_weight * directau_loss
            ).detach().float().item(),
            "opsd_weighted_token_loss": (
                self.args.opsd_token_weight * token_loss
            ).detach().float().item(),
            "opsd_weighted_item_loss": (
                rank_weight * item_loss
            ).detach().float().item(),
            "lightgcn_weighted_rank_loss": (
                lightgcn_weight * lightgcn_loss
            ).detach().float().item(),
            "sasrec_weighted_rank_loss": (
                sasrec_weight * sasrec_loss
            ).detach().float().item(),
            "opsd_weighted_small_moe_loss": (
                small_moe_weight * small_moe_loss
            ).detach().float().item(),
            "opsd_recommendation_space_loss": (
                recommendation_space_loss.detach().float().item()
            ),
            "opsd_text_space_loss": (
                text_space_loss.detach().float().item()
            ),
            "opsd_total_loss": total_loss.detach().float().item(),
        }
        if rationale_geometry_weight != 0:
            metrics["opsd_rationale_geometry_loss"] = (
                rationale_geometry_loss.detach().float().item()
            )
            metrics["opsd_weighted_rationale_geometry_loss"] = (
                rationale_geometry_weight * rationale_geometry_loss
            ).detach().float().item()
        if target_loss_type == "full_softmax_ce":
            metrics.pop("opsd_target_infonce_loss")
            metrics.pop("opsd_weighted_target_infonce_loss")
            metrics.pop("opsd_target_infonce_gradient_scale")
            metrics["opsd_target_full_softmax_ce_loss"] = (
                target_loss.detach().float().item()
            )
            metrics["opsd_weighted_target_full_softmax_ce_loss"] = (
                target_weight * target_loss
            ).detach().float().item()
        elif target_loss_type == "privileged_directau":
            metrics.pop("opsd_target_infonce_loss")
            metrics.pop("opsd_weighted_target_infonce_loss")
            metrics.pop("opsd_target_infonce_gradient_scale")
            metrics["opsd_target_privileged_directau_loss"] = (
                target_loss.detach().float().item()
            )
            metrics["opsd_weighted_target_privileged_directau_loss"] = (
                target_weight * target_loss
            ).detach().float().item()
        if getattr(self, "_opsd_frozen_teacher_model", None) is not None:
            current_step = int(self.state.global_step)
            metrics.update({
                "opsd_teacher_sync_count": float(
                    self._opsd_teacher_sync_count
                ),
                "opsd_teacher_age_steps": float(max(
                    0,
                    current_step - self._opsd_teacher_last_sync_step,
                )),
                "opsd_teacher_last_update_norm": float(
                    self._opsd_teacher_last_update_norm
                ),
            })
        metrics.update({
            key: value.detach().float().item()
            for key, value in recommendation_metrics.items()
        })
        self.store_metrics(metrics, metric_key_prefix="train")
        return total_loss
