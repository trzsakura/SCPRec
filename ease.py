"""Standalone EASE training and frozen full-catalog teacher for Recommendation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from sasrec import PAD_ITEM_ID, EvalRow, build_eval_rows, build_train_data
from expert_validation import split_expert_training_rows


def _best_validation_quality(training_info: Mapping[str, Any] | None) -> float:
    info = dict(training_info or {})
    metrics = info.get("best_metrics")
    if not isinstance(metrics, Mapping):
        return 1.0
    value = metrics.get("ndcg@10")
    if value is None:
        values = [
            metric
            for key, metric in metrics.items()
            if str(key).lower().startswith("ndcg@")
        ]
        value = values[0] if values else None
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return 1.0
    return quality if math.isfinite(quality) and quality > 0 else 1.0


@torch.no_grad()
def fit_ease_weights(
    user_positive_item_ids: Sequence[Sequence[int]],
    n_items: int,
    regularization: float = 500.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Fit the closed-form EASE item-item matrix, excluding padding item 0."""

    if n_items <= 2:
        raise ValueError("EASE requires padding plus at least two real items.")
    if not math.isfinite(float(regularization)) or regularization <= 0:
        raise ValueError("EASE regularization must be finite and positive.")
    resolved_device = torch.device(device)
    real_items = n_items - 1
    gram = torch.zeros(
        (real_items, real_items),
        dtype=torch.float32,
        device=resolved_device,
    )
    valid_users = 0
    for raw_items in user_positive_item_ids:
        unique = sorted({int(item_id) for item_id in raw_items if int(item_id) != 0})
        if any(item_id < 1 or item_id >= n_items for item_id in unique):
            raise ValueError("An EASE training item is outside the real item catalog.")
        if not unique:
            continue
        valid_users += 1
        indices = torch.tensor(unique, dtype=torch.long, device=resolved_device) - 1
        gram[indices.unsqueeze(1), indices.unsqueeze(0)] += 1.0
    if valid_users == 0:
        raise ValueError("EASE received no positive user-item interactions.")
    gram.diagonal().add_(float(regularization))
    precision = torch.linalg.inv(gram)
    diagonal = precision.diagonal()
    if (~torch.isfinite(diagonal)).any() or diagonal.abs().min() <= 1e-12:
        raise ValueError("EASE inverse has an invalid diagonal; increase regularization.")
    weights = -precision / diagonal.unsqueeze(0)
    weights.fill_diagonal_(0.0)
    if (~torch.isfinite(weights)).any():
        raise ValueError("EASE fitted non-finite item-item weights.")
    return weights.detach().cpu()


@torch.no_grad()
def save_ease_checkpoint(
    output_path: str | Path,
    weights: torch.Tensor,
    train_item_mask: torch.BoolTensor,
    regularization: float,
    training_info: Mapping[str, Any] | None = None,
) -> None:
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("EASE weights must be a square matrix.")
    n_items = int(weights.shape[0]) + 1
    mask = train_item_mask.detach().reshape(-1).bool().cpu()
    if mask.shape[0] != n_items:
        raise ValueError("EASE train item mask does not match the weight matrix.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "ease_id",
            "weights": weights.detach().float().cpu(),
            "train_item_mask": mask,
            "n_items": n_items,
            "regularization": float(regularization),
            "training_info": dict(training_info or {}),
        },
        output_path,
    )


def _torch_load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("EASE checkpoint must contain a dictionary.")
    return checkpoint


@dataclass
class EASETeacher:
    weights: torch.Tensor
    train_item_mask: torch.BoolTensor
    source_path: str
    regularization: float
    validation_quality: float

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        expected_num_items: int | None = None,
    ) -> "EASETeacher":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EASE checkpoint does not exist: {checkpoint_path}")
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        required = {"weights", "train_item_mask", "regularization"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"EASE checkpoint is missing {sorted(missing)}.")
        if checkpoint.get("model_type") not in (None, "ease_id"):
            raise ValueError(
                f"Unsupported EASE checkpoint type: {checkpoint.get('model_type')!r}."
            )
        weights = checkpoint["weights"].detach().float().cpu()
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("EASE checkpoint weights must be square.")
        n_items = int(weights.shape[0]) + 1
        stored_n_items = int(checkpoint.get("n_items", n_items))
        if stored_n_items != n_items:
            raise ValueError("EASE checkpoint n_items disagrees with its weights.")
        if expected_num_items is not None and n_items != expected_num_items:
            raise ValueError(
                "EASE/Recommendation item catalog mismatch: "
                f"checkpoint={n_items}, Recommendation={expected_num_items}."
            )
        mask = checkpoint["train_item_mask"].detach().reshape(-1).bool().cpu()
        if mask.shape[0] != n_items or bool(mask[PAD_ITEM_ID]) or not mask.any():
            raise ValueError("EASE train item mask is invalid.")
        return cls(
            weights=weights,
            train_item_mask=mask,
            source_path=str(checkpoint_path),
            regularization=float(checkpoint["regularization"]),
            validation_quality=_best_validation_quality(
                checkpoint.get("training_info")
            ),
        )

    @property
    def n_items(self) -> int:
        return int(self.weights.shape[0]) + 1

    @torch.no_grad()
    def score_histories(self, history_item_ids: torch.LongTensor) -> torch.Tensor:
        """Sum learned rows for all unique visible history items."""

        if history_item_ids.ndim != 2:
            raise ValueError("EASE histories must have shape [batch, sequence].")
        histories = history_item_ids.detach().long().cpu()
        if histories.numel() and (
            (histories < 0).any() or (histories >= self.n_items).any()
        ):
            raise ValueError("An EASE history item is outside the item catalog.")
        scores = torch.zeros(
            (histories.shape[0], self.n_items),
            dtype=self.weights.dtype,
        )
        for row_index, row in enumerate(histories):
            items = torch.unique(row[(row > PAD_ITEM_ID)])
            if items.numel():
                scores[row_index, 1:] = self.weights[items - 1].sum(dim=0)
        return scores


@torch.no_grad()
def evaluate_ease(
    teacher: EASETeacher,
    rows: Sequence[EvalRow],
    k: int = 10,
    batch_size: int = 256,
) -> dict[str, float]:
    if k <= 0 or batch_size <= 0:
        raise ValueError("Evaluation k and batch_size must be positive.")
    if not rows:
        return {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0, "eval_rows": 0.0}
    hit_total = 0.0
    ndcg_total = 0.0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        max_width = max(1, max(len(row.history_item_ids) for row in batch))
        histories = torch.zeros((len(batch), max_width), dtype=torch.long)
        for index, row in enumerate(batch):
            if row.history_item_ids:
                visible = torch.tensor(row.history_item_ids, dtype=torch.long)
                histories[index, -visible.numel() :] = visible
        scores = teacher.score_histories(histories)
        scores[:, ~teacher.train_item_mask] = -torch.inf
        for index, row in enumerate(batch):
            if row.history_item_ids:
                scores[index, list(row.history_item_ids)] = -torch.inf
        targets = torch.tensor([row.target_item_id for row in batch], dtype=torch.long)
        target_scores = scores.gather(1, targets.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1) + 1
        hits = ranks <= k
        hit_total += hits.float().sum().item()
        ndcg_total += torch.where(
            hits,
            1.0 / torch.log2(ranks.float() + 1.0),
            torch.zeros_like(ranks, dtype=torch.float),
        ).sum().item()
    denominator = float(len(rows))
    return {
        f"hr@{k}": hit_total / denominator,
        f"ndcg@{k}": ndcg_total / denominator,
        "eval_rows": denominator,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit EASE on an Recommendation DatasetDict.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--regularization", type=float, default=500.0)
    parser.add_argument("--fit_device", default="cpu")
    parser.add_argument("--valid_k", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    from datasets import disable_progress_bars, load_from_disk

    disable_progress_bars()
    dataset_dict = load_from_disk(args.dataset_dir)
    fit_dataset, expert_validation = split_expert_training_rows(
        dataset_dict,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_data = build_train_data(fit_dataset)
    positives_by_user: dict[str, frozenset[int]] = {}
    for example in train_data.examples:
        positives_by_user[example.user_id] = example.known_positive_item_ids
    weights = fit_ease_weights(
        tuple(positives_by_user.values()),
        n_items=train_data.n_items,
        regularization=args.regularization,
        device=args.fit_device,
    )
    provisional = EASETeacher(
        weights=weights,
        train_item_mask=train_data.train_item_mask.cpu(),
        source_path="<in-memory>",
        regularization=args.regularization,
        validation_quality=1.0,
    )
    valid_rows, valid_cold = build_eval_rows(
        expert_validation,
        train_data.n_items,
        train_item_mask=train_data.train_item_mask,
    )
    best_metrics = evaluate_ease(
        provisional,
        valid_rows,
        k=args.valid_k,
        batch_size=args.eval_batch_size,
    )
    final_metrics = {}
    for split_name in ("valid", "test"):
        rows, cold_items = build_eval_rows(
            dataset_dict[split_name],
            train_data.n_items,
            train_item_mask=train_data.train_item_mask,
        )
        metrics = evaluate_ease(
            provisional,
            rows,
            k=args.valid_k,
            batch_size=args.eval_batch_size,
        )
        final_metrics[split_name] = {
            **metrics,
            "cold_items": int(cold_items),
        }
        print(
            f"[{split_name}] hr@{args.valid_k}="
            f"{metrics[f'hr@{args.valid_k}']:.6f} "
            f"ndcg@{args.valid_k}="
            f"{metrics[f'ndcg@{args.valid_k}']:.6f} "
            f"rows={int(metrics['eval_rows'])} cold_items={cold_items}"
        )
    training_info = {
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "dataset_dir": str(Path(args.dataset_dir)),
        "seed": args.seed,
        "validation_source": "train",
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.seed,
        "validation_rows": len(expert_validation),
        "fit_rows": len(fit_dataset["train"]),
        "valid_cold_items": valid_cold,
        "train_users": len(positives_by_user),
    }
    save_ease_checkpoint(
        args.output_path,
        weights,
        train_data.train_item_mask,
        args.regularization,
        training_info,
    )
    print(f"Saved EASE checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
