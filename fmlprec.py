"""Standalone FMLP-Rec training and frozen-teacher utilities for Recommendation.

The implementation intentionally owns only the sequential recommender model.
It reuses Recommendation dataset parsing/evaluation helpers from :mod:`sasrec`, but it
does not depend on the existing standalone SASRec distillation loss.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sasrec import (
    PAD_ITEM_ID,
    EvalRow,
    SASRecExample,
    SASRecTrainData,
    _sample_negative_items,
    build_eval_rows,
    build_train_data,
    evaluate_sasrec,
    prepare_history_batch,
)
from expert_validation import split_expert_training_rows


def _best_validation_quality(training_info: Mapping[str, Any] | None) -> float:
    """Extract a positive validation-quality prior without assuming one cutoff."""

    info = dict(training_info or {})
    metrics = info.get("best_metrics")
    if not isinstance(metrics, Mapping):
        return 1.0
    preferred = metrics.get("ndcg@10")
    if preferred is None:
        ndcg_values = [
            value
            for key, value in metrics.items()
            if str(key).lower().startswith("ndcg@")
        ]
        preferred = ndcg_values[0] if ndcg_values else None
    try:
        quality = float(preferred)
    except (TypeError, ValueError):
        return 1.0
    return quality if math.isfinite(quality) and quality > 0 else 1.0


class FMLPFilterLayer(nn.Module):
    """Learnable complex frequency filter over the sequence dimension."""

    def __init__(self, hidden_dim: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        frequency_bins = max_seq_len // 2 + 1
        self.max_seq_len = int(max_seq_len)
        self.complex_weight = nn.Parameter(
            torch.empty(frequency_bins, hidden_dim, 2)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        nn.init.normal_(self.complex_weight, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        residual = hidden
        frequency = torch.fft.rfft(hidden.float(), dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight.float())
        filtered = torch.fft.irfft(
            frequency * weight.unsqueeze(0),
            n=self.max_seq_len,
            dim=1,
            norm="ortho",
        ).to(hidden.dtype)
        hidden = self.norm(residual + self.dropout(filtered))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class FMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.filter = FMLPFilterLayer(hidden_dim, max_seq_len, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-8)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        hidden = self.filter(hidden, padding_mask)
        hidden = self.norm(hidden + self.ffn(hidden))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class FMLPRec(nn.Module):
    """Filter-enhanced MLP sequential recommender with tied item embeddings."""

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_items <= 2:
            raise ValueError("FMLP-Rec requires padding plus at least two real items.")
        if hidden_dim <= 0 or num_blocks <= 0 or max_seq_len <= 0:
            raise ValueError("hidden_dim, num_blocks, and max_seq_len must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.n_items = int(n_items)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.max_seq_len = int(max_seq_len)
        self.dropout_rate = float(dropout)
        self.item_embedding = nn.Embedding(
            self.n_items,
            self.hidden_dim,
            padding_idx=PAD_ITEM_ID,
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.start_embedding = nn.Parameter(torch.empty(self.hidden_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            FMLPBlock(self.hidden_dim, self.max_seq_len, dropout)
            for _ in range(self.num_blocks)
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-8)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.start_embedding, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[PAD_ITEM_ID].zero_()

    def _encode_prepared(self, item_sequences: torch.LongTensor) -> torch.Tensor:
        if item_sequences.ndim != 2 or item_sequences.shape[1] != self.max_seq_len:
            raise ValueError(
                "Prepared FMLP-Rec sequences must have shape [batch, max_seq_len]."
            )
        if item_sequences.numel() and (
            (item_sequences < 0).any() or (item_sequences >= self.n_items).any()
        ):
            raise ValueError("An FMLP-Rec history item is outside the item catalog.")
        padding_mask = item_sequences.eq(PAD_ITEM_ID)
        empty_rows = padding_mask.all(dim=1)
        safe_padding_mask = padding_mask.clone()
        if empty_rows.any():
            safe_padding_mask[empty_rows, -1] = False
        positions = torch.arange(
            self.max_seq_len,
            dtype=torch.long,
            device=item_sequences.device,
        ).unsqueeze(0)
        hidden = self.item_embedding(item_sequences) * math.sqrt(self.hidden_dim)
        hidden = hidden + self.position_embedding(positions)
        if empty_rows.any():
            hidden[empty_rows, -1] = (
                self.start_embedding
                + self.position_embedding.weight[self.max_seq_len - 1]
            )
        hidden = self.embedding_dropout(hidden)
        hidden = hidden.masked_fill(safe_padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, safe_padding_mask)
        hidden = self.final_norm(hidden)
        hidden = hidden.masked_fill(safe_padding_mask.unsqueeze(-1), 0.0)
        return hidden[:, -1]

    def encode(self, item_sequences: torch.LongTensor) -> torch.Tensor:
        prepared = prepare_history_batch(
            item_sequences,
            self.max_seq_len,
            device=item_sequences.device,
        )
        return self._encode_prepared(prepared)

    def score_all_items(self, item_sequences: torch.LongTensor) -> torch.Tensor:
        return self.encode(item_sequences) @ self.item_embedding.weight.t()

    def score_prepared_items(self, item_sequences: torch.LongTensor) -> torch.Tensor:
        return self._encode_prepared(item_sequences) @ self.item_embedding.weight.t()

    def sampled_bce_loss(
        self,
        item_sequences: torch.LongTensor,
        positive_item_ids: torch.LongTensor,
        negative_item_ids: torch.LongTensor,
    ) -> torch.Tensor:
        user_embeddings = self._encode_prepared(item_sequences)
        positive_item_ids = positive_item_ids.reshape(-1).to(item_sequences.device)
        negative_item_ids = negative_item_ids.to(item_sequences.device)
        if negative_item_ids.ndim == 1:
            negative_item_ids = negative_item_ids.unsqueeze(1)
        if positive_item_ids.shape[0] != user_embeddings.shape[0] or (
            negative_item_ids.ndim != 2
            or negative_item_ids.shape[0] != user_embeddings.shape[0]
        ):
            raise ValueError("FMLP-Rec target/negative batch does not match histories.")
        positive = self.item_embedding(positive_item_ids)
        negative = self.item_embedding(negative_item_ids)
        positive_logits = (user_embeddings * positive).sum(dim=-1)
        negative_logits = torch.einsum("bd,bnd->bn", user_embeddings, negative)
        return F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()


def train_fmlprec(
    train_data: SASRecTrainData,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    max_seq_len: int = 50,
    dropout: float = 0.2,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_negatives: int = 1,
    weight_decay: float = 0.0,
    max_grad_norm: float = 5.0,
    device: str | torch.device = "cuda",
    seed: int = 42,
    valid_rows: Sequence[EvalRow] | None = None,
    valid_k: int = 10,
    eval_every: int = 5,
    patience: int = 10,
    eval_batch_size: int = 256,
) -> tuple[FMLPRec, dict[str, Any]]:
    if epochs <= 0 or batch_size <= 0 or lr <= 0 or num_negatives <= 0:
        raise ValueError("epochs, batch_size, lr, and num_negatives must be positive.")
    resolved_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = FMLPRec(
        train_data.n_items,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    candidates = torch.where(train_data.train_item_mask)[0].numpy()
    examples = train_data.examples
    best_state = None
    best_metrics = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(len(examples))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            selected = permutation[start : start + batch_size]
            batch_examples: list[SASRecExample] = [
                examples[int(index)] for index in selected
            ]
            histories = prepare_history_batch(
                [example.history_item_ids for example in batch_examples],
                max_seq_len,
                device=resolved_device,
            )
            targets = torch.tensor(
                [example.target_item_id for example in batch_examples],
                dtype=torch.long,
                device=resolved_device,
            )
            negatives = _sample_negative_items(
                batch_examples,
                candidates,
                num_negatives,
                rng,
                resolved_device,
            )
            loss = model.sampled_bce_loss(histories, targets, negatives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                model.item_embedding.weight[PAD_ITEM_ID].zero_()
            total_loss += loss.detach().item()
            n_batches += 1
        final_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} - FMLP-Rec Loss: {final_loss:.6f}")
        if valid_rows and (epoch % eval_every == 0 or epoch == epochs):
            metrics = evaluate_sasrec(
                model,
                valid_rows,
                k=valid_k,
                batch_size=eval_batch_size,
                candidate_item_mask=train_data.train_item_mask,
            )
            score = metrics[f"ndcg@{valid_k}"]
            print(
                f"  [valid] hr@{valid_k}={metrics[f'hr@{valid_k}']:.6f} "
                f"ndcg@{valid_k}={score:.6f} rows={int(metrics['eval_rows'])}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f"Early stopping after {patience} validation checks.")
                    break
    if best_state is not None:
        model.load_state_dict(
            {key: value.to(resolved_device) for key, value in best_state.items()}
        )
    else:
        best_epoch = epoch
    return model, {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_fmlprec_loss": final_loss,
    }


@torch.no_grad()
def save_fmlprec_checkpoint(
    output_path: str | Path,
    model: FMLPRec,
    train_item_mask: torch.BoolTensor,
    training_info: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "fmlprec_id",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "train_item_mask": train_item_mask.detach().reshape(-1).bool().cpu(),
            "n_items": model.n_items,
            "hidden_dim": model.hidden_dim,
            "num_blocks": model.num_blocks,
            "max_seq_len": model.max_seq_len,
            "dropout": model.dropout_rate,
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
        raise TypeError("FMLP-Rec checkpoint must contain a dictionary.")
    return checkpoint


@dataclass
class FMLPRecTeacher:
    model: FMLPRec
    train_item_mask: torch.BoolTensor
    source_path: str
    validation_quality: float

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        expected_num_items: int | None = None,
    ) -> "FMLPRecTeacher":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FMLP-Rec checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        required = {
            "model_state_dict",
            "train_item_mask",
            "n_items",
            "hidden_dim",
            "num_blocks",
            "max_seq_len",
            "dropout",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"FMLP-Rec checkpoint is missing {sorted(missing)}.")
        if checkpoint.get("model_type") not in (None, "fmlprec_id"):
            raise ValueError(
                f"Unsupported FMLP-Rec checkpoint type: {checkpoint.get('model_type')!r}."
            )
        n_items = int(checkpoint["n_items"])
        if expected_num_items is not None and n_items != expected_num_items:
            raise ValueError(
                "FMLP-Rec/Recommendation item catalog mismatch: "
                f"checkpoint={n_items}, Recommendation={expected_num_items}."
            )
        model = FMLPRec(
            n_items=n_items,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_blocks=int(checkpoint["num_blocks"]),
            max_seq_len=int(checkpoint["max_seq_len"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.requires_grad_(False)
        model.eval()
        mask = checkpoint["train_item_mask"].detach().reshape(-1).bool()
        if mask.shape[0] != n_items or bool(mask[PAD_ITEM_ID]) or not mask.any():
            raise ValueError("FMLP-Rec train item mask is invalid.")
        return cls(
            model=model,
            train_item_mask=mask,
            source_path=str(checkpoint_path),
            validation_quality=_best_validation_quality(
                checkpoint.get("training_info")
            ),
        )

    def to(self, device: str | torch.device) -> "FMLPRecTeacher":
        self.model = self.model.to(device)
        self.model.eval()
        self.train_item_mask = self.train_item_mask.to(device)
        return self

    @torch.no_grad()
    def score_histories(self, history_item_ids: torch.LongTensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return self.model.score_all_items(history_item_ids.to(device))


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train FMLP-Rec on an Recommendation DatasetDict.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--valid_k", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--device", default=None)
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
    valid_rows, valid_cold = build_eval_rows(
        expert_validation,
        train_data.n_items,
        train_item_mask=train_data.train_item_mask,
    )
    model, training_info = train_fmlprec(
        train_data=train_data,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=_resolve_device(args.device),
        seed=args.seed,
        valid_rows=valid_rows,
        valid_k=args.valid_k,
        eval_every=args.eval_every,
        patience=args.patience,
        eval_batch_size=args.eval_batch_size,
    )
    training_info.update(
        {
            "dataset_dir": str(Path(args.dataset_dir)),
            "seed": args.seed,
            "validation_source": "train",
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.seed,
            "validation_rows": len(expert_validation),
            "fit_rows": len(fit_dataset["train"]),
            "valid_cold_items": valid_cold,
        }
    )
    final_metrics = {}
    for split_name in ("valid", "test"):
        rows, cold_items = build_eval_rows(
            dataset_dict[split_name],
            train_data.n_items,
            train_item_mask=train_data.train_item_mask,
        )
        metrics = evaluate_sasrec(
            model,
            rows,
            k=args.valid_k,
            batch_size=args.eval_batch_size,
            candidate_item_mask=train_data.train_item_mask,
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
    training_info["final_metrics"] = final_metrics
    save_fmlprec_checkpoint(
        args.output_path,
        model,
        train_data.train_item_mask,
        training_info,
    )
    print(f"Saved FMLP-Rec checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
