"""Independent four-expert recommendation teacher MoE for OPSD.

The pool is deliberately fixed to SASRec, LightGCN, FMLP-Rec, and EASE.
Routing uses a selectable per-sample advantage, while each expert's held-out
NDCG supplies a static reliability prior.  This module does not reuse or
modify the existing standalone SASRec/LightGCN distillation losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ease import EASETeacher
from fmlprec import FMLPRecTeacher
from lightgcn import LightGCNEmbeddingBank
from sasrec import SASRecTeacher


REQUIRED_SMALL_MOE_EXPERTS = ("sasrec", "lightgcn", "fmlprec", "ease")
SMALL_MOE_ROUTER_MODES = ("soft_mixture", "hard_best")
SMALL_MOE_ADVANTAGE_MODES = ("target_rank", "candidate_bpr")


def normalize_small_moe_router_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SMALL_MOE_ROUTER_MODES:
        raise ValueError(
            "small_moe_router_mode must be one of: "
            + ", ".join(SMALL_MOE_ROUTER_MODES)
        )
    return normalized


def normalize_small_moe_advantage_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "normalized_log_rank":
        normalized = "target_rank"
    if normalized not in SMALL_MOE_ADVANTAGE_MODES:
        raise ValueError(
            "small_moe_advantage_mode must be one of: "
            + ", ".join(SMALL_MOE_ADVANTAGE_MODES)
        )
    return normalized


def _torch_load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Small-teacher checkpoint must be a dictionary: {path}")
    return checkpoint


def _checkpoint_validation_quality(path: str | Path) -> float:
    checkpoint = _torch_load_checkpoint(path)
    training_info = checkpoint.get("training_info")
    if not isinstance(training_info, Mapping):
        raise ValueError(
            f"MoE checkpoint {path} is missing training_info with validation NDCG."
        )
    metrics = training_info.get("best_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"MoE checkpoint {path} is missing best_metrics with validation NDCG."
        )
    value = metrics.get("ndcg@10")
    if value is None:
        values = [
            metric
            for key, metric in metrics.items()
            if str(key).lower().startswith("ndcg@")
        ]
        value = values[0] if values else None
    if value is None:
        raise ValueError(
            f"MoE checkpoint {path} best_metrics has no validation NDCG."
        )
    try:
        quality = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"MoE checkpoint {path} has a non-numeric validation NDCG: {value!r}."
        ) from error
    if not math.isfinite(quality) or quality < 0.0 or quality > 1.0:
        raise ValueError(
            f"MoE checkpoint {path} validation NDCG must be in [0, 1]; "
            f"got {quality!r}."
        )
    # A measured zero is valid evidence, unlike a missing metric. Keep the
    # prior positive so its logarithm remains finite in the router.
    return max(quality, 1e-8)


@dataclass(frozen=True)
class SmallTeacherOutput:
    name: str
    scores: torch.Tensor
    valid_item_mask: torch.BoolTensor
    valid_example_mask: torch.BoolTensor
    validation_quality: float


def _masked_mean(values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    if mask.any():
        return values[mask].float().mean()
    return values.float().sum() * 0.0


def _candidate_bpr_loss(
    scores: torch.Tensor,
    target_index: torch.LongTensor,
    negative_rows: torch.LongTensor,
    negative_items: torch.LongTensor,
    negative_count: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Detached per-example BPR on the shared candidate negatives only."""

    target_scores = scores.gather(1, target_index).squeeze(1)
    margins = (
        scores[negative_rows, negative_items] - target_scores[negative_rows]
    ) / float(temperature)
    sums = torch.zeros_like(target_scores).scatter_add_(
        0, negative_rows, F.softplus(margins)
    )
    return sums / negative_count.clamp_min(1).to(sums.dtype)


def _candidate_distribution_metrics(
    probabilities: torch.Tensor,
    target_index: torch.LongTensor,
    negative_mask: torch.BoolTensor,
    valid: torch.BoolTensor,
) -> dict[str, torch.Tensor]:
    negative_count = negative_mask.sum(dim=1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=1)
    target_probability = probabilities.gather(1, target_index).squeeze(1)
    top_negative_probability = probabilities.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=1).values
    top_negative_probability = torch.where(
        negative_count.gt(0),
        top_negative_probability,
        torch.zeros_like(top_negative_probability),
    )
    return {
        "entropy": _masked_mean(entropy, valid),
        "target_prob": _masked_mean(target_probability, valid),
        "target_top_negative_prob_gap": _masked_mean(
            target_probability - top_negative_probability,
            valid & negative_count.gt(0),
        ),
    }


def small_teacher_moe_distillation_loss(
    student_scores: torch.Tensor,
    teacher_outputs: Mapping[str, SmallTeacherOutput],
    history_item_ids: torch.LongTensor,
    target_item_ids: torch.LongTensor,
    topk: int = 32,
    student_temperature: float = 0.2,
    teacher_temperatures: Mapping[str, float] | None = None,
    gate_temperature: float = 0.1,
    outer_temperature: float = 0.1,
    quality_power: float = 1.0,
    router_mode: str = "soft_mixture",
    advantage_mode: str = "target_rank",
    return_per_example: bool = False,
):
    """Distil an advantage-routed mixture of four frozen recommenders.

    Only experts better than the current student under ``advantage_mode``
    receive router mass. ``soft_mixture`` normalizes mass across all
    such experts; ``hard_best`` selects only the largest-advantage expert and
    uses validation quality solely to break equal-advantage ties. A separate
    outer gate preserves how much better the best expert actually is.
    """

    if tuple(sorted(teacher_outputs)) != tuple(sorted(REQUIRED_SMALL_MOE_EXPERTS)):
        raise ValueError(
            "small-teacher MoE requires exactly these experts: "
            + ", ".join(REQUIRED_SMALL_MOE_EXPERTS)
        )
    if student_scores.ndim != 2:
        raise ValueError("student_scores must have shape [batch, items].")
    batch_size, n_items = student_scores.shape
    if n_items <= 1:
        raise ValueError("Small-teacher MoE needs at least two catalog items.")
    if history_item_ids.ndim != 2 or history_item_ids.shape[0] != batch_size:
        raise ValueError("history_item_ids must have shape [batch, history].")
    target_item_ids = target_item_ids.reshape(-1).to(student_scores.device)
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("target_item_ids does not match the score batch.")
    if ((target_item_ids <= 0) | (target_item_ids >= n_items)).any():
        raise ValueError("A small-teacher MoE target is outside the real catalog.")
    if topk < 0:
        raise ValueError("small_moe_topk must be non-negative; use 0 for full catalog.")
    scalars = {
        "student_temperature": student_temperature,
        "gate_temperature": gate_temperature,
        "outer_temperature": outer_temperature,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(float(quality_power)) or float(quality_power) < 0:
        raise ValueError("quality_power must be finite and non-negative.")
    router_mode = normalize_small_moe_router_mode(router_mode)
    advantage_mode = normalize_small_moe_advantage_mode(advantage_mode)
    temperatures = dict(teacher_temperatures or {})
    for name in REQUIRED_SMALL_MOE_EXPERTS:
        temperature = float(temperatures.get(name, 0.2))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Teacher temperature for {name} must be positive.")
        temperatures[name] = temperature

    device = student_scores.device
    common_items = torch.ones(n_items, dtype=torch.bool, device=device)
    detached_scores: dict[str, torch.Tensor] = {}
    valid_examples: dict[str, torch.BoolTensor] = {}
    qualities: dict[str, float] = {}
    for name in REQUIRED_SMALL_MOE_EXPERTS:
        output = teacher_outputs[name]
        if output.name != name:
            raise ValueError(f"Small-teacher output {name!r} has name {output.name!r}.")
        scores = output.scores.detach().to(device=device, dtype=torch.float32)
        if scores.shape != student_scores.shape:
            raise ValueError(f"{name} scores do not match student full-catalog scores.")
        item_mask = output.valid_item_mask.detach().reshape(-1).to(
            device=device,
            dtype=torch.bool,
        )
        if item_mask.shape[0] != n_items:
            raise ValueError(f"{name} valid_item_mask has the wrong catalog size.")
        item_mask = item_mask.clone()
        item_mask[0] = False
        common_items &= item_mask
        example_mask = output.valid_example_mask.detach().reshape(-1).to(
            device=device,
            dtype=torch.bool,
        )
        if example_mask.shape[0] != batch_size:
            raise ValueError(f"{name} valid_example_mask has the wrong batch size.")
        quality = float(output.validation_quality)
        if not math.isfinite(quality) or quality <= 0:
            quality = 1e-8
        detached_scores[name] = scores
        valid_examples[name] = example_mask
        qualities[name] = quality
    if not common_items.any():
        raise ValueError("The four MoE teachers have no common train-observed items.")

    histories = history_item_ids.to(device=device, dtype=torch.long)
    available = common_items.unsqueeze(0).expand(batch_size, -1).clone()
    valid_history = (histories > 0) & (histories < n_items)
    if valid_history.any():
        rows = torch.arange(device=device, end=batch_size).unsqueeze(1).expand_as(
            histories
        )
        available[rows[valid_history], histories[valid_history]] = False
    target_index = target_item_ids.unsqueeze(1)
    target_available = available.gather(1, target_index).squeeze(1)
    available_count = available.sum(dim=1).float()
    log_catalog_size = available_count.clamp_min(2.0).log()

    student_float = student_scores.float()
    student_rank_scores = student_float.detach().masked_fill(~available, -torch.inf)
    student_target_score = student_rank_scores.gather(1, target_index)
    student_rank = (student_rank_scores > student_target_score).sum(dim=1) + 1
    student_reward = (
        1.0 - student_rank.float().log() / log_catalog_size
    ).clamp(0.0, 1.0)

    # Construct the same candidate union used by the KL before computing a
    # Candidate-BPR advantage. The paper-default target-rank route uses dense
    # full-catalog ranks instead.
    if topk == 0:
        candidate_mask = available.clone()
    else:
        candidate_mask = torch.zeros_like(available)
        k = min(int(topk), max(n_items - 1, 1))
        student_topk = torch.topk(student_rank_scores, k=k, dim=1).indices
        candidate_mask.scatter_(1, student_topk, True)
        for name in REQUIRED_SMALL_MOE_EXPERTS:
            teacher_rank_scores = detached_scores[name].masked_fill(
                ~available, -torch.inf,
            )
            teacher_topk = torch.topk(teacher_rank_scores, k=k, dim=1).indices
            candidate_mask.scatter_(1, teacher_topk, True)
        candidate_mask.scatter_(1, target_index, True)
        candidate_mask &= available
    # Inactive rows can have unavailable targets (e.g. a catalog-cold sample).
    # Add one harmless slot so their softmax is defined; outer_strength is zero.
    empty_candidates = ~candidate_mask.any(dim=1)
    if empty_candidates.any():
        fallback = common_items.nonzero(as_tuple=False)[0, 0]
        candidate_mask[empty_candidates, fallback] = True
    negative_mask = candidate_mask.clone()
    negative_mask.scatter_(1, target_index, False)
    negative_rows, negative_items = negative_mask.nonzero(as_tuple=True)
    negative_count = negative_mask.sum(dim=1)
    with torch.no_grad():
        student_bpr = _candidate_bpr_loss(
            student_float.detach(), target_index, negative_rows,
            negative_items, negative_count, student_temperature,
        )

    teacher_ranks: dict[str, torch.Tensor] = {}
    teacher_bpr_losses: dict[str, torch.Tensor] = {}
    teacher_utilities: dict[str, torch.Tensor] = {}
    expert_valid: dict[str, torch.BoolTensor] = {}
    advantages: dict[str, torch.Tensor] = {}
    expert_active: dict[str, torch.BoolTensor] = {}
    student_utility = (
        -student_bpr if advantage_mode == "candidate_bpr" else student_reward
    ).detach()
    for name in REQUIRED_SMALL_MOE_EXPERTS:
        rank_scores = detached_scores[name].masked_fill(~available, -torch.inf)
        target_score = rank_scores.gather(1, target_index)
        rank = (rank_scores > target_score).sum(dim=1) + 1
        reward = (1.0 - rank.float().log() / log_catalog_size).clamp(0.0, 1.0)
        valid = valid_examples[name] & target_available
        with torch.no_grad():
            teacher_bpr = _candidate_bpr_loss(
                detached_scores[name], target_index, negative_rows,
                negative_items, negative_count, temperatures[name],
            )
        teacher_bpr_losses[name] = teacher_bpr
        if advantage_mode == "candidate_bpr":
            advantage = (student_bpr - teacher_bpr).clamp_min(0.0)
            valid &= negative_count.gt(0)
            utility = -teacher_bpr
        else:
            advantage = (reward - student_reward).clamp_min(0.0)
            utility = reward
        advantage = advantage * valid.to(advantage.dtype)
        teacher_ranks[name] = rank
        advantages[name] = advantage.detach()
        teacher_utilities[name] = utility.detach()
        expert_valid[name] = valid.detach()
        expert_active[name] = valid & advantage.gt(0)

    advantage_matrix = torch.stack(
        [advantages[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
        dim=1,
    )
    active_matrix = torch.stack(
        [expert_active[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
        dim=1,
    )
    quality_prior = torch.tensor(
        [qualities[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-8)
    router_logits = (
        advantage_matrix / float(gate_temperature)
        + float(quality_power) * quality_prior.log().unsqueeze(0)
    )
    router_logits = router_logits.masked_fill(~active_matrix, -torch.inf)
    any_active = active_matrix.any(dim=1)
    maximum_advantage = advantage_matrix.max(dim=1).values
    best_advantage_ties = active_matrix & advantage_matrix.eq(
        maximum_advantage.unsqueeze(1)
    )
    tie_count = best_advantage_ties.sum(dim=1)
    selected_matrix = torch.zeros_like(active_matrix)
    if router_mode == "soft_mixture":
        safe_logits = router_logits.clone()
        safe_logits[~any_active] = 0.0
        router_weights = F.softmax(safe_logits, dim=1)
        router_weights = router_weights * active_matrix.to(router_weights.dtype)
        router_weights = router_weights / router_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
    else:
        # Sample quality determines the winner. Validation NDCG is only a
        # deterministic reliability tie-breaker, and quality_power=0 disables
        # that prior so the fixed REQUIRED_SMALL_MOE_EXPERTS order wins ties.
        tie_break_logits = (
            float(quality_power) * quality_prior.log().unsqueeze(0)
        ).expand(batch_size, -1)
        tie_break_logits = tie_break_logits.masked_fill(
            ~best_advantage_ties,
            -torch.inf,
        )
        safe_tie_break_logits = tie_break_logits.clone()
        safe_tie_break_logits[~any_active] = 0.0
        selected_indices = safe_tie_break_logits.argmax(dim=1)
        selected_matrix = F.one_hot(
            selected_indices,
            num_classes=len(REQUIRED_SMALL_MOE_EXPERTS),
        ).bool()
        selected_matrix &= any_active.unsqueeze(1)
        router_weights = selected_matrix.to(dtype=advantage_matrix.dtype)
    router_weights = router_weights.detach()
    outer_strength = (
        1.0 - torch.exp(-maximum_advantage / float(outer_temperature))
    ).detach()

    fill_value = -1e4
    student_logits = (student_float / float(student_temperature)).masked_fill(
        ~candidate_mask,
        fill_value,
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    mixture = torch.zeros_like(student_log_probs)
    routed_teacher_entropy = torch.zeros(batch_size, device=device)
    distribution_metrics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        student_stats = _candidate_distribution_metrics(
            student_log_probs.detach().exp(), target_index,
            negative_mask, target_available,
        )
        distribution_metrics.update({
            f"small_moe_student_{key}": value
            for key, value in student_stats.items()
        })
    for expert_index, name in enumerate(REQUIRED_SMALL_MOE_EXPERTS):
        teacher_logits = (
            detached_scores[name] / temperatures[name]
        ).masked_fill(~candidate_mask, fill_value)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        mixture = mixture + router_weights[:, expert_index : expert_index + 1] * teacher_probs
        with torch.no_grad():
            teacher_entropy = -(
                teacher_probs * teacher_probs.clamp_min(1e-12).log()
            ).sum(dim=1)
            routed_teacher_entropy += (
                router_weights[:, expert_index] * teacher_entropy
            )
            teacher_stats = _candidate_distribution_metrics(
                teacher_probs, target_index, negative_mask,
                valid_examples[name] & target_available,
            )
            distribution_metrics.update({
                f"small_moe_{name}_{key}": value
                for key, value in teacher_stats.items()
            })
    mixture = mixture.detach()
    with torch.no_grad():
        mixture_stats = _candidate_distribution_metrics(
            mixture, target_index, negative_mask,
            any_active & target_available,
        )
        distribution_metrics.update({
            f"small_moe_mixture_{key}": value
            for key, value in mixture_stats.items()
        })
        mixture_entropy = -(
            mixture * mixture.clamp_min(1e-12).log()
        ).sum(dim=1)
        distribution_metrics["small_moe_router_weighted_teacher_entropy"] = (
            _masked_mean(routed_teacher_entropy, any_active)
        )
        distribution_metrics["small_moe_mixture_entropy_excess"] = (
            _masked_mean(
                mixture_entropy - routed_teacher_entropy, any_active,
            )
        )
    per_example_kl = (
        mixture
        * (mixture.clamp_min(1e-12).log() - student_log_probs)
    ).sum(dim=1)
    loss = (outer_strength * per_example_kl).mean()

    metrics: dict[str, torch.Tensor] = {
        "small_moe_active_rate": any_active.float().mean(),
        "small_moe_outer_strength": outer_strength.mean(),
        "small_moe_max_advantage": maximum_advantage.mean(),
        "small_moe_student_target_rank": student_rank.float().mean(),
        "small_moe_candidate_count": candidate_mask.sum(dim=1).float().mean(),
        "small_moe_common_item_count": common_items.float().sum(),
        "small_moe_bpr_negative_count": _masked_mean(
            negative_count, target_available,
        ),
        "small_moe_student_candidate_bpr_loss": _masked_mean(
            student_bpr, target_available & negative_count.gt(0),
        ),
        "small_moe_candidate_bpr_enabled": student_scores.new_tensor(
            float(advantage_mode == "candidate_bpr")
        ),
        "small_moe_hard_best_enabled": student_scores.new_tensor(
            float(router_mode == "hard_best")
        ),
        "small_moe_best_expert_tie_rate": (
            (any_active & tie_count.gt(1)).float().mean()
        ),
    }
    metrics.update(distribution_metrics)
    for expert_index, name in enumerate(REQUIRED_SMALL_MOE_EXPERTS):
        valid = valid_examples[name] & target_available
        metrics.update(
            {
                f"small_moe_{name}_router_weight": router_weights[
                    :, expert_index
                ].mean(),
                f"small_moe_{name}_positive_advantage_rate": expert_active[
                    name
                ].float().mean(),
                f"small_moe_{name}_advantage": _masked_mean(
                    advantages[name],
                    valid,
                ),
                f"small_moe_{name}_candidate_bpr_loss": _masked_mean(
                    teacher_bpr_losses[name], valid & negative_count.gt(0),
                ),
                f"small_moe_{name}_target_rank": _masked_mean(
                    teacher_ranks[name],
                    valid,
                ),
                f"small_moe_{name}_valid_rate": valid.float().mean(),
                f"small_moe_{name}_validation_quality": student_scores.new_tensor(
                    qualities[name]
                ),
            }
        )
        if router_mode == "hard_best":
            metrics[f"small_moe_{name}_selected_rate"] = selected_matrix[
                :, expert_index
            ].float().mean()
    if not return_per_example:
        return loss, metrics

    expert_valid_matrix = torch.stack(
        [expert_valid[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
        dim=1,
    )
    utility_matrix = torch.stack(
        [teacher_utilities[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
        dim=1,
    )
    masked_utilities = utility_matrix.masked_fill(
        ~expert_valid_matrix,
        -torch.inf,
    )
    any_valid_expert = expert_valid_matrix.any(dim=1)
    best_utility = masked_utilities.max(dim=1).values
    best_utility_ties = (
        expert_valid_matrix
        & masked_utilities.eq(best_utility.unsqueeze(1))
        & any_valid_expert.unsqueeze(1)
    )
    per_example = {
        "expert_names": REQUIRED_SMALL_MOE_EXPERTS,
        "advantage_mode": advantage_mode,
        "router_mode": router_mode,
        "target_available": target_available.detach(),
        "candidate_count": candidate_mask.sum(dim=1).detach(),
        "negative_count": negative_count.detach(),
        "student_rank": student_rank.detach(),
        "student_candidate_bpr_loss": student_bpr.detach(),
        "student_utility": student_utility,
        "expert_ranks": torch.stack(
            [teacher_ranks[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
            dim=1,
        ).detach(),
        "expert_candidate_bpr_losses": torch.stack(
            [teacher_bpr_losses[name] for name in REQUIRED_SMALL_MOE_EXPERTS],
            dim=1,
        ).detach(),
        "expert_utilities": utility_matrix.detach(),
        "expert_valid": expert_valid_matrix.detach(),
        "advantages": advantage_matrix.detach(),
        "expert_active": active_matrix.detach(),
        "any_active": any_active.detach(),
        "router_weights": router_weights.detach(),
        "outer_strength": outer_strength.detach(),
        "best_utility_ties": best_utility_ties.detach(),
        "best_utility_tie_count": best_utility_ties.sum(dim=1).detach(),
    }
    return loss, metrics, per_example


@dataclass
class SmallTeacherMoE:
    sasrec: SASRecTeacher
    lightgcn: LightGCNEmbeddingBank
    fmlprec: FMLPRecTeacher
    ease: EASETeacher
    validation_qualities: dict[str, float]

    @classmethod
    def from_checkpoints(
        cls,
        sasrec_checkpoint: str | Path,
        lightgcn_checkpoint: str | Path,
        fmlprec_checkpoint: str | Path,
        ease_checkpoint: str | Path,
        expected_num_items: int,
    ) -> "SmallTeacherMoE":
        paths = {
            "sasrec": sasrec_checkpoint,
            "lightgcn": lightgcn_checkpoint,
            "fmlprec": fmlprec_checkpoint,
            "ease": ease_checkpoint,
        }
        qualities = {
            name: _checkpoint_validation_quality(path)
            for name, path in paths.items()
        }
        return cls(
            sasrec=SASRecTeacher.from_checkpoint(
                sasrec_checkpoint,
                expected_num_items=expected_num_items,
            ),
            lightgcn=LightGCNEmbeddingBank.from_checkpoint(
                lightgcn_checkpoint,
                expected_num_items=expected_num_items,
            ),
            fmlprec=FMLPRecTeacher.from_checkpoint(
                fmlprec_checkpoint,
                expected_num_items=expected_num_items,
            ),
            ease=EASETeacher.from_checkpoint(
                ease_checkpoint,
                expected_num_items=expected_num_items,
            ),
            validation_qualities=qualities,
        )

    def to(self, device: str | torch.device) -> "SmallTeacherMoE":
        """Move neural/embedding experts to training device; keep EASE on CPU."""

        self.sasrec.to(device)
        self.lightgcn.to(device)
        self.fmlprec.to(device)
        self.ease.weights = self.ease.weights.cpu()
        self.ease.train_item_mask = self.ease.train_item_mask.cpu()
        return self

    @torch.no_grad()
    def score(
        self,
        history_item_ids: torch.LongTensor,
        user_ids: Sequence[str],
        student_device: str | torch.device,
    ) -> dict[str, SmallTeacherOutput]:
        device = torch.device(student_device)
        batch_size = history_item_ids.shape[0]
        all_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        sasrec_scores = self.sasrec.score_histories(history_item_ids).to(device)
        lightgcn_scores, lightgcn_valid = self.lightgcn.score_users(user_ids)
        fmlprec_scores = self.fmlprec.score_histories(history_item_ids).to(device)
        ease_scores = self.ease.score_histories(history_item_ids).to(device)
        return {
            "sasrec": SmallTeacherOutput(
                name="sasrec",
                scores=sasrec_scores,
                valid_item_mask=self.sasrec.train_item_mask.to(device),
                valid_example_mask=all_valid,
                validation_quality=self.validation_qualities["sasrec"],
            ),
            "lightgcn": SmallTeacherOutput(
                name="lightgcn",
                scores=lightgcn_scores.to(device),
                valid_item_mask=self.lightgcn.train_item_mask.to(device),
                valid_example_mask=lightgcn_valid.to(device),
                validation_quality=self.validation_qualities["lightgcn"],
            ),
            "fmlprec": SmallTeacherOutput(
                name="fmlprec",
                scores=fmlprec_scores,
                valid_item_mask=self.fmlprec.train_item_mask.to(device),
                valid_example_mask=all_valid,
                validation_quality=self.validation_qualities["fmlprec"],
            ),
            "ease": SmallTeacherOutput(
                name="ease",
                scores=ease_scores,
                valid_item_mask=self.ease.train_item_mask.to(device),
                valid_example_mask=all_valid,
                validation_quality=self.validation_qualities["ease"],
            ),
        }
