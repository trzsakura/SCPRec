"""Standalone SASRec training and OPSD sequential-rank utilities.

The script consumes the same on-disk Recommendation ``DatasetDict`` as ``lightgcn.py``.
Only the train split is used to fit SASRec.  Validation and test rows keep their
own ordered visible histories, and their targets never enter model fitting or
the train-observed candidate catalog.

Example::

    python sasrec.py \
        --dataset_dir data/Amazon_Books_CoLLM_PositiveOnly_UserSample25K_Dedup \
        --output_path checkpoints/sasrec_amazon_book_25k.pt \
        --device cuda:7

The saved checkpoint can also be loaded by OPSD as a frozen, history-dependent
ranking teacher.
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

from expert_validation import split_expert_training_rows


PAD_ITEM_ID = 0
_SASREC_TEACHER_GATE_MODES = {"hard", "advantage_weighted"}
_SASREC_TEACHER_REWARD_MODES = {
    "candidate_bpr",
    "normalized_log_rank",
}


def _normalize_sasrec_teacher_gate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _SASREC_TEACHER_GATE_MODES:
        choices = ", ".join(sorted(_SASREC_TEACHER_GATE_MODES))
        raise ValueError(
            f"sasrec_teacher_gate_mode must be one of: {choices}."
        )
    return normalized


def _normalize_sasrec_teacher_reward_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _SASREC_TEACHER_REWARD_MODES:
        choices = ", ".join(sorted(_SASREC_TEACHER_REWARD_MODES))
        raise ValueError(
            f"sasrec_teacher_reward_mode must be one of: {choices}."
        )
    return normalized


def _split_columns(split: Any) -> tuple[str, str]:
    column_names = set(split.column_names)
    history_column = (
        "history_item_id"
        if "history_item_id" in column_names
        else "seq_input_ids"
    )
    target_column = "item_id" if "item_id" in column_names else "seq_labels"
    missing = {
        name
        for name in (history_column, target_column)
        if name not in column_names
    }
    if missing:
        raise KeyError(f"Recommendation split is missing required columns: {sorted(missing)}")
    return history_column, target_column


def _validate_item_catalog(item_split: Any) -> int:
    if "item_id" not in item_split.column_names:
        raise KeyError("item_info must contain item_id.")
    item_ids = sorted({int(item_id) for item_id in item_split["item_id"]})
    if not item_ids or item_ids[0] != PAD_ITEM_ID:
        raise ValueError("Recommendation item catalog must contain padding item id 0.")
    expected = list(range(item_ids[-1] + 1))
    if item_ids != expected:
        raise ValueError(
            "SASRec requires dense item IDs aligned with Recommendation's item bank; "
            f"found {len(item_ids)} IDs spanning 0..{item_ids[-1]}."
        )
    return len(expected)


def _validate_positive_labels(split: Any, split_name: str) -> None:
    if "label" not in set(split.column_names):
        return
    non_positive = sum(int(label) != 1 for label in split["label"])
    if non_positive:
        raise ValueError(
            f"SASRec treats every {split_name} row as a positive next-item "
            f"event, but found {non_positive} rows with label != 1. Build or "
            "select a positive-only Recommendation dataset first."
        )


@dataclass(frozen=True)
class SASRecExample:
    user_id: str
    history_item_ids: tuple[int, ...]
    target_item_id: int
    known_positive_item_ids: frozenset[int]


@dataclass(frozen=True)
class SASRecTrainData:
    examples: tuple[SASRecExample, ...]
    n_items: int
    train_item_mask: torch.BoolTensor
    n_users: int


@dataclass(frozen=True)
class EvalRow:
    history_item_ids: tuple[int, ...]
    target_item_id: int


def _clean_history(history: Sequence[Any], n_items: int) -> tuple[int, ...]:
    cleaned: list[int] = []
    for raw_item_id in history:
        item_id = int(raw_item_id)
        if item_id == PAD_ITEM_ID:
            continue
        if item_id < 0 or item_id >= n_items:
            raise ValueError(
                f"History item id {item_id} is outside catalog range "
                f"0..{n_items - 1}."
            )
        cleaned.append(item_id)
    return tuple(cleaned)


def build_train_data(dataset_dict: Mapping[str, Any]) -> SASRecTrainData:
    """Build leakage-free next-item examples from the train split only."""

    if "train" not in dataset_dict or "item_info" not in dataset_dict:
        raise KeyError("DatasetDict must contain train and item_info splits.")
    train_split = dataset_dict["train"]
    _validate_positive_labels(train_split, "train")
    history_column, target_column = _split_columns(train_split)
    if "user_id" not in set(train_split.column_names):
        raise KeyError("Recommendation train split is missing required column: user_id")
    n_items = _validate_item_catalog(dataset_dict["item_info"])

    raw_rows: list[tuple[str, tuple[int, ...], int]] = []
    positives_by_user: dict[str, set[int]] = {}
    train_item_mask = torch.zeros(n_items, dtype=torch.bool)
    for raw_user_id, raw_history, raw_target in zip(
        train_split["user_id"],
        train_split[history_column],
        train_split[target_column],
    ):
        user_id = str(raw_user_id)
        history = _clean_history(raw_history, n_items)
        target = int(raw_target)
        if target <= PAD_ITEM_ID or target >= n_items:
            raise ValueError(
                f"Train target item id {target} is outside real catalog range "
                f"1..{n_items - 1}."
            )
        raw_rows.append((user_id, history, target))
        user_positives = positives_by_user.setdefault(user_id, set())
        user_positives.update(history)
        user_positives.add(target)
        if history:
            train_item_mask[list(history)] = True
        train_item_mask[target] = True

    if not raw_rows:
        raise ValueError("No positive train interactions were found for SASRec.")
    train_item_mask[PAD_ITEM_ID] = False
    if int(train_item_mask.sum()) < 2:
        raise ValueError("SASRec negative sampling requires at least two train items.")
    frozen_positives_by_user = {
        user_id: frozenset(item_ids)
        for user_id, item_ids in positives_by_user.items()
    }
    examples = tuple(
        SASRecExample(
            user_id=user_id,
            history_item_ids=history,
            target_item_id=target,
            known_positive_item_ids=frozen_positives_by_user[user_id],
        )
        for user_id, history, target in raw_rows
    )
    return SASRecTrainData(
        examples=examples,
        n_items=n_items,
        train_item_mask=train_item_mask,
        n_users=len(positives_by_user),
    )


def build_eval_rows(
    split: Any,
    n_items: int,
    train_item_mask: torch.BoolTensor | None = None,
) -> tuple[list[EvalRow], int]:
    """Convert an untouched evaluation split and count unavailable targets."""

    _validate_positive_labels(split, "evaluation")
    history_column, target_column = _split_columns(split)
    if train_item_mask is not None:
        train_item_mask = train_item_mask.detach().reshape(-1).bool().cpu()
        if train_item_mask.shape[0] != n_items:
            raise ValueError("train_item_mask does not match the item catalog.")
    rows: list[EvalRow] = []
    unavailable_items = 0
    for raw_history, raw_target in zip(
        split[history_column],
        split[target_column],
    ):
        target = int(raw_target)
        target_is_unavailable = (
            target <= PAD_ITEM_ID
            or target >= n_items
            or (
                train_item_mask is not None
                and not bool(train_item_mask[target])
            )
        )
        if target_is_unavailable:
            unavailable_items += 1
            continue
        history = _clean_history(raw_history, n_items)
        if target in history:
            raise ValueError(
                f"Evaluation target {target} already occurs in its visible history."
            )
        rows.append(EvalRow(history_item_ids=history, target_item_id=target))
    return rows, unavailable_items


def prepare_history_batch(
    histories: Sequence[Sequence[int]] | torch.Tensor,
    max_seq_len: int,
    device: str | torch.device | None = None,
) -> torch.LongTensor:
    """Compact padding, keep recent items, and right-align every history."""

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if isinstance(histories, torch.Tensor):
        if histories.ndim != 2:
            raise ValueError("History tensor must have shape [batch, sequence].")
        resolved_device = (
            histories.device if device is None else torch.device(device)
        )
        source = histories.detach().to(device=resolved_device, dtype=torch.long)
        prepared = torch.zeros(
            (source.shape[0], max_seq_len),
            dtype=torch.long,
            device=resolved_device,
        )
        valid = source.ne(PAD_ITEM_ID)
        if valid.any():
            source_rank = valid.long().cumsum(dim=1) - 1
            lengths = valid.long().sum(dim=1)
            source_start = (lengths - max_seq_len).clamp_min(0)
            kept = valid & source_rank.ge(source_start.unsqueeze(1))
            kept_lengths = lengths.clamp_max(max_seq_len)
            destination = (
                max_seq_len
                - kept_lengths.unsqueeze(1)
                + source_rank
                - source_start.unsqueeze(1)
            )
            row_indices = torch.arange(
                source.shape[0],
                device=resolved_device,
            ).unsqueeze(1).expand_as(source)
            prepared[row_indices[kept], destination[kept]] = source[kept]
        return prepared

    source_rows = histories
    resolved_device = torch.device(device) if device is not None else None
    prepared = torch.zeros(
        (len(source_rows), max_seq_len),
        dtype=torch.long,
        device=resolved_device,
    )
    for row_index, raw_history in enumerate(source_rows):
        compact = [int(item_id) for item_id in raw_history if int(item_id) != 0]
        compact = compact[-max_seq_len:]
        if compact:
            prepared[row_index, -len(compact) :] = torch.tensor(
                compact,
                dtype=torch.long,
                device=resolved_device,
            )
    return prepared


class SASRecBlock(nn.Module):
    """Pre-normalized causal self-attention block used by SASRec."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.BoolTensor,
        causal_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        query = self.attention_norm(hidden)
        attention_output, _ = self.attention(
            query,
            hidden,
            hidden,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = query + attention_output
        normalized = self.ffn_norm(hidden)
        hidden = normalized + self.ffn(normalized)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation with tied item embeddings."""

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_items <= 2:
            raise ValueError("SASRec requires padding plus at least two real items.")
        if hidden_dim <= 0 or num_blocks <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, num_blocks, and num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_items = int(n_items)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.dropout_rate = float(dropout)
        self.item_embedding = nn.Embedding(
            self.n_items,
            self.hidden_dim,
            padding_idx=PAD_ITEM_ID,
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_len,
            self.hidden_dim,
        )
        self.start_embedding = nn.Parameter(torch.empty(self.hidden_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(self.hidden_dim, self.num_heads, dropout)
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
        if item_sequences.ndim != 2:
            raise ValueError("item_sequences must have shape [batch, sequence].")
        if item_sequences.shape[1] != self.max_seq_len:
            raise ValueError(
                "Prepared SASRec sequences must have width max_seq_len="
                f"{self.max_seq_len}."
            )
        if item_sequences.numel() and (
            (item_sequences < 0).any() or (item_sequences >= self.n_items).any()
        ):
            raise ValueError("A SASRec history item is outside the item catalog.")

        padding_mask = item_sequences.eq(PAD_ITEM_ID)
        empty_rows = padding_mask.all(dim=1)
        safe_padding_mask = padding_mask.clone()
        if empty_rows.any():
            safe_padding_mask[empty_rows, -1] = False

        positions = torch.arange(
            self.max_seq_len,
            device=item_sequences.device,
            dtype=torch.long,
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
        causal_mask = torch.triu(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                device=item_sequences.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, safe_padding_mask, causal_mask)
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
        user_embeddings = self.encode(item_sequences)
        return user_embeddings @ self.item_embedding.weight.t()

    def score_prepared_items(
        self,
        item_sequences: torch.LongTensor,
    ) -> torch.Tensor:
        """Score histories already compacted to exactly ``max_seq_len``."""

        user_embeddings = self._encode_prepared(item_sequences)
        return user_embeddings @ self.item_embedding.weight.t()

    def sampled_bce_loss(
        self,
        item_sequences: torch.LongTensor,
        positive_item_ids: torch.LongTensor,
        negative_item_ids: torch.LongTensor,
    ) -> torch.Tensor:
        user_embeddings = self._encode_prepared(item_sequences)
        positive_item_ids = positive_item_ids.reshape(-1).to(
            item_sequences.device
        )
        if positive_item_ids.shape[0] != user_embeddings.shape[0]:
            raise ValueError("Positive item batch does not match histories.")
        negative_item_ids = negative_item_ids.to(item_sequences.device)
        if negative_item_ids.ndim == 1:
            negative_item_ids = negative_item_ids.unsqueeze(1)
        if (
            negative_item_ids.ndim != 2
            or negative_item_ids.shape[0] != user_embeddings.shape[0]
        ):
            raise ValueError("Negative items must have shape [batch, negatives].")
        positive_embeddings = self.item_embedding(positive_item_ids)
        negative_embeddings = self.item_embedding(negative_item_ids)
        positive_logits = (user_embeddings * positive_embeddings).sum(dim=-1)
        negative_logits = torch.einsum(
            "bd,bnd->bn",
            user_embeddings,
            negative_embeddings,
        )
        return F.softplus(-positive_logits).mean() + F.softplus(
            negative_logits
        ).mean()


def _sample_negative_items(
    examples: Sequence[SASRecExample],
    candidate_item_ids: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.LongTensor:
    if num_negatives <= 0:
        raise ValueError("num_negatives must be positive.")
    candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
    if candidate_item_ids.ndim != 1 or candidate_item_ids.size == 0:
        raise ValueError("Negative sampling requires train-observed item IDs.")
    negatives: list[list[int]] = []
    for example in examples:
        positives = example.known_positive_item_ids
        if len(positives) >= candidate_item_ids.size:
            raise ValueError(
                f"User {example.user_id!r} has no train-observed negative item."
            )
        sampled: list[int] = []
        while len(sampled) < num_negatives:
            item_id = int(
                candidate_item_ids[rng.integers(0, candidate_item_ids.size)]
            )
            if item_id not in positives:
                sampled.append(item_id)
        negatives.append(sampled)
    return torch.tensor(negatives, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_sasrec(
    model: SASRec,
    rows: Sequence[EvalRow],
    k: int = 10,
    batch_size: int = 256,
    candidate_item_mask: torch.BoolTensor | None = None,
) -> dict[str, float]:
    """Exact full-catalog HR/NDCG for one target per Recommendation row."""

    if k <= 0 or batch_size <= 0:
        raise ValueError("Evaluation k and batch_size must be positive.")
    if not rows:
        return {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0, "eval_rows": 0.0}
    device = next(model.parameters()).device
    if candidate_item_mask is None:
        candidate_item_mask = torch.ones(model.n_items, dtype=torch.bool)
        candidate_item_mask[PAD_ITEM_ID] = False
    candidate_item_mask = candidate_item_mask.detach().reshape(-1).to(
        device=device,
        dtype=torch.bool,
    )
    if candidate_item_mask.shape[0] != model.n_items:
        raise ValueError("candidate_item_mask does not match SASRec items.")
    candidate_item_mask = candidate_item_mask.clone()
    candidate_item_mask[PAD_ITEM_ID] = False

    was_training = model.training
    model.eval()
    hit_total = 0.0
    ndcg_total = 0.0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        histories = prepare_history_batch(
            [row.history_item_ids for row in batch_rows],
            model.max_seq_len,
            device=device,
        )
        scores = model.score_prepared_items(histories)
        scores[:, ~candidate_item_mask] = -torch.inf
        for row_index, row in enumerate(batch_rows):
            if row.history_item_ids:
                history = torch.tensor(
                    row.history_item_ids,
                    dtype=torch.long,
                    device=device,
                )
                scores[row_index, history] = -torch.inf
        targets = torch.tensor(
            [row.target_item_id for row in batch_rows],
            dtype=torch.long,
            device=device,
        )
        if (~candidate_item_mask[targets]).any():
            raise ValueError("An evaluation target is not train-observed.")
        target_scores = scores.gather(1, targets.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1) + 1
        hits = ranks <= k
        hit_total += hits.float().sum().item()
        ndcg_total += torch.where(
            hits,
            1.0 / torch.log2(ranks.float() + 1.0),
            torch.zeros_like(ranks, dtype=torch.float),
        ).sum().item()
    model.train(was_training)
    denominator = float(len(rows))
    return {
        f"hr@{k}": hit_total / denominator,
        f"ndcg@{k}": ndcg_total / denominator,
        "eval_rows": denominator,
    }


def train_sasrec(
    train_data: SASRecTrainData,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    num_heads: int = 2,
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
) -> tuple[SASRec, dict[str, Any]]:
    """Train SASRec with sampled binary cross-entropy."""

    if epochs <= 0 or batch_size <= 0 or lr <= 0 or num_negatives <= 0:
        raise ValueError(
            "epochs, batch_size, lr, and num_negatives must be positive."
        )
    if weight_decay < 0 or max_grad_norm < 0:
        raise ValueError("weight_decay and max_grad_norm must be non-negative.")
    if eval_every <= 0 or patience < 0 or eval_batch_size <= 0:
        raise ValueError(
            "eval_every/eval_batch_size must be positive and patience non-negative."
        )

    resolved_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    model = SASRec(
        n_items=train_data.n_items,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    candidate_item_ids = torch.where(train_data.train_item_mask)[0].numpy()
    examples = train_data.examples

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -float("inf")
    best_epoch = 0
    evaluations_without_improvement = 0
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(len(examples))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_examples = [examples[int(index)] for index in batch_indices]
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
                candidate_item_ids,
                num_negatives,
                rng,
                resolved_device,
            )
            loss = model.sampled_bce_loss(histories, targets, negatives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
            optimizer.step()
            with torch.no_grad():
                model.item_embedding.weight[PAD_ITEM_ID].zero_()
            total_loss += loss.detach().item()
            n_batches += 1

        final_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} - SASRec Loss: {final_loss:.6f}")
        should_evaluate = bool(valid_rows) and (
            epoch % eval_every == 0 or epoch == epochs
        )
        if should_evaluate:
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
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if patience and evaluations_without_improvement >= patience:
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
        "final_sasrec_loss": final_loss,
    }


@torch.no_grad()
def save_sasrec_checkpoint(
    output_path: str | Path,
    model: SASRec,
    train_data: SASRecTrainData,
    training_info: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "sasrec_id",
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "train_item_mask": train_data.train_item_mask.detach().cpu(),
            "n_items": model.n_items,
            "hidden_dim": model.hidden_dim,
            "num_blocks": model.num_blocks,
            "num_heads": model.num_heads,
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
        raise TypeError("SASRec checkpoint must contain a dictionary.")
    return checkpoint


@dataclass
class SASRecTeacher:
    model: SASRec
    train_item_mask: torch.BoolTensor
    source_path: str

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        expected_num_items: int | None = None,
    ) -> "SASRecTeacher":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SASRec checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        required = {
            "model_state_dict",
            "train_item_mask",
            "n_items",
            "hidden_dim",
            "num_blocks",
            "num_heads",
            "max_seq_len",
            "dropout",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(
                "Checkpoint is missing Recommendation SASRec artifacts "
                f"{sorted(missing)}. Train it with this repository's sasrec.py."
            )
        if checkpoint.get("model_type") not in (None, "sasrec_id"):
            raise ValueError(
                f"Unsupported SASRec checkpoint type: {checkpoint.get('model_type')!r}."
            )
        n_items = int(checkpoint["n_items"])
        if expected_num_items is not None and n_items != expected_num_items:
            raise ValueError(
                "SASRec/Recommendation item catalog mismatch: "
                f"checkpoint={n_items}, Recommendation={expected_num_items}."
            )
        model = SASRec(
            n_items=n_items,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_blocks=int(checkpoint["num_blocks"]),
            num_heads=int(checkpoint["num_heads"]),
            max_seq_len=int(checkpoint["max_seq_len"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.requires_grad_(False)
        model.eval()
        train_item_mask = checkpoint["train_item_mask"].detach().reshape(-1).bool()
        if train_item_mask.shape[0] != n_items:
            raise ValueError("SASRec train item mask has the wrong size.")
        if bool(train_item_mask[PAD_ITEM_ID]):
            raise ValueError("Padding item 0 cannot be train-observed.")
        if not train_item_mask.any():
            raise ValueError("SASRec checkpoint has no train-observed items.")
        return cls(
            model=model,
            train_item_mask=train_item_mask,
            source_path=str(checkpoint_path),
        )

    def to(self, device: str | torch.device) -> "SASRecTeacher":
        self.model = self.model.to(device)
        self.model.eval()
        self.train_item_mask = self.train_item_mask.to(device)
        return self

    @torch.no_grad()
    def score_histories(self, history_item_ids: torch.LongTensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return self.model.score_all_items(history_item_ids.to(device))


def _masked_mean(values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    if mask.any():
        return values[mask].float().mean()
    return values.float().sum() * 0.0


def sasrec_rank_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    history_item_ids: torch.LongTensor,
    target_item_ids: torch.LongTensor,
    valid_item_mask: torch.BoolTensor | None = None,
    topk: int = 32,
    teacher_temperature: float = 0.2,
    student_temperature: float = 0.2,
    teacher_better_only: bool = True,
    teacher_gate_mode: str = "hard",
    teacher_gate_temperature: float = 0.1,
    teacher_gate_power: float = 1.0,
    teacher_reward_mode: str = "normalized_log_rank",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distil SASRec ranking on target/teacher-top-k/student-top-k union."""

    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError("Student/SASRec scores must share shape [batch, items].")
    batch_size, n_items = student_scores.shape
    if n_items <= 1:
        raise ValueError("Sequential rank distillation needs at least two items.")
    if history_item_ids.ndim != 2 or history_item_ids.shape[0] != batch_size:
        raise ValueError("history_item_ids must have shape [batch, history].")
    target_item_ids = target_item_ids.reshape(-1).to(student_scores.device)
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("Target IDs do not match score batch size.")
    if topk < 0:
        raise ValueError("sasrec_topk must be non-negative; use 0 for full catalog.")
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("SASRec distillation temperatures must be positive.")
    teacher_gate_mode = _normalize_sasrec_teacher_gate_mode(
        teacher_gate_mode
    )
    teacher_reward_mode = _normalize_sasrec_teacher_reward_mode(
        teacher_reward_mode
    )
    teacher_gate_temperature = float(teacher_gate_temperature)
    teacher_gate_power = float(teacher_gate_power)
    if (
        not math.isfinite(teacher_gate_temperature)
        or teacher_gate_temperature <= 0
    ):
        raise ValueError(
            "sasrec_teacher_gate_temperature must be finite and positive."
        )
    if (
        not math.isfinite(teacher_gate_power)
        or teacher_gate_power <= 0
    ):
        raise ValueError(
            "sasrec_teacher_gate_power must be finite and positive."
        )
    if ((target_item_ids <= PAD_ITEM_ID) | (target_item_ids >= n_items)).any():
        raise ValueError("A SASRec target item is outside the real item catalog.")

    history_item_ids = history_item_ids.to(student_scores.device)
    if valid_item_mask is None:
        valid_item_mask = torch.ones(
            n_items,
            dtype=torch.bool,
            device=student_scores.device,
        )
    else:
        valid_item_mask = valid_item_mask.reshape(-1).to(
            device=student_scores.device,
            dtype=torch.bool,
        )
        if valid_item_mask.shape[0] != n_items:
            raise ValueError("valid_item_mask does not match the score catalog.")
    valid_item_mask = valid_item_mask.clone()
    valid_item_mask[PAD_ITEM_ID] = False
    available = valid_item_mask.unsqueeze(0).expand(batch_size, -1).clone()
    valid_history = (
        (history_item_ids > PAD_ITEM_ID) & (history_item_ids < n_items)
    )
    if valid_history.any():
        rows = torch.arange(
            batch_size,
            device=student_scores.device,
        ).unsqueeze(1).expand_as(history_item_ids)
        available[rows[valid_history], history_item_ids[valid_history]] = False
    if (~available.gather(1, target_item_ids.unsqueeze(1))).any():
        raise ValueError(
            "A SASRec target is unavailable, padding, or already in history."
        )

    student_float = student_scores.float()
    teacher_float = teacher_scores.detach().to(student_scores.device).float()
    student_full_masked = student_float.detach().masked_fill(~available, -torch.inf)
    teacher_full_masked = teacher_float.masked_fill(~available, -torch.inf)
    target_index = target_item_ids.unsqueeze(1)
    student_target_score = student_full_masked.gather(1, target_index)
    teacher_target_score = teacher_full_masked.gather(1, target_index)
    student_rank = (student_full_masked > student_target_score).sum(dim=1) + 1
    teacher_rank = (teacher_full_masked > teacher_target_score).sum(dim=1) + 1

    if topk == 0:
        candidate_mask = available
    else:
        k = min(topk, n_items - 1)
        teacher_topk = torch.topk(teacher_full_masked, k=k, dim=1).indices
        student_topk = torch.topk(student_full_masked, k=k, dim=1).indices
        candidate_mask = torch.zeros_like(available)
        candidate_mask.scatter_(1, teacher_topk, True)
        candidate_mask.scatter_(1, student_topk, True)
        candidate_mask.scatter_(1, target_index, True)
        candidate_mask &= available

    fill_value = -1e4
    student_logits = (student_float / student_temperature).masked_fill(
        ~candidate_mask,
        fill_value,
    )
    teacher_logits = (teacher_float / teacher_temperature).masked_fill(
        ~candidate_mask,
        fill_value,
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_target_probability = student_log_probs.gather(
        1,
        target_index,
    ).exp().squeeze(1)
    teacher_target_probability = teacher_log_probs.gather(
        1,
        target_index,
    ).exp().squeeze(1)

    # student_reward = (
    #     (1.0 - reward_softmax_weight) * student_ndcg_reward
    #     + reward_softmax_weight * student_target_probability
    # )
    # teacher_reward = (
    #     (1.0 - reward_softmax_weight) * teacher_ndcg_reward
    #     + reward_softmax_weight * teacher_target_probability
    # )
    if teacher_reward_mode == "normalized_log_rank":
        # A dense catalog-normalized rank reward.  Unlike the legacy mixed
        # reward, this does not depend on a cutoff, candidate-union softmax,
        # or teacher/student score calibration.  Clamp N to two for the
        # degenerate one-available-item case, where both ranks must be one.
        available_item_count = available.sum(dim=1).float()
        log_catalog_size = available_item_count.clamp_min(2.0).log()
        student_reward = 1.0 - student_rank.float().log() / log_catalog_size
        teacher_reward = 1.0 - teacher_rank.float().log() / log_catalog_size
        student_reward = student_reward.clamp(0.0, 1.0)
        teacher_reward = teacher_reward.clamp(0.0, 1.0)
    else:
        # Compare target-vs-negative margins on the exact same candidate
        # union used by the KL. The temperatures are the model-specific score
        # calibration already used above. Reward is negative BPR loss, so the
        # teacher advantage equals L_bpr(student) - L_bpr(teacher).
        negative_mask = candidate_mask.clone()
        negative_mask.scatter_(1, target_index, False)
        negative_count = negative_mask.sum(dim=1)
        safe_negative_count = negative_count.clamp_min(1).float()

        student_scaled_scores = student_float / student_temperature
        teacher_scaled_scores = teacher_float / teacher_temperature
        student_target_scaled = student_scaled_scores.gather(1, target_index)
        teacher_target_scaled = teacher_scaled_scores.gather(1, target_index)
        student_pairwise_loss = F.softplus(
            student_scaled_scores - student_target_scaled
        ).masked_fill(~negative_mask, 0.0)
        teacher_pairwise_loss = F.softplus(
            teacher_scaled_scores - teacher_target_scaled
        ).masked_fill(~negative_mask, 0.0)
        student_bpr_loss = (
            student_pairwise_loss.sum(dim=1) / safe_negative_count
        )
        teacher_bpr_loss = (
            teacher_pairwise_loss.sum(dim=1) / safe_negative_count
        )
        student_reward = -student_bpr_loss
        teacher_reward = -teacher_bpr_loss
    reward_gap = (teacher_reward - student_reward).detach()

    if not bool(teacher_better_only):
        gate_weights = torch.ones_like(reward_gap)
    elif teacher_gate_mode == "hard":
        gate_weights = (teacher_rank < student_rank).to(reward_gap.dtype)
    else:
        gate_weights = (
            1.0
            - torch.exp(
                -reward_gap.clamp_min(0.0) / teacher_gate_temperature
            )
        ).pow(teacher_gate_power)
    gate_weights = gate_weights.detach()
    distill_mask = gate_weights > 0

    per_example_kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    weight_sum = gate_weights.sum()
    safe_weight_sum = torch.where(
        weight_sum > 0,
        weight_sum,
        torch.ones_like(weight_sum),
    )
    weighted_kl_sum = (per_example_kl * gate_weights).sum()
    if bool(teacher_better_only) and teacher_gate_mode == "advantage_weighted":
        # Preserve the absolute amount of positive teacher advantage instead
        # of renormalizing it away.  Batch-mean reduction keeps the loss
        # independent of batch size, while the fixed 10x factor keeps the
        # typical ~0.1 soft weights on the legacy SASRec-loss scale.
        # Hard/no-gate retain their legacy reductions for compatibility.
        loss = 10.0 * weighted_kl_sum / batch_size
    else:
        loss = weighted_kl_sum / safe_weight_sum

    all_examples = torch.ones_like(distill_mask)
    positive_gap = reward_gap > 0
    metric_zero = reward_gap[:0].sum()
    positive_gap_mean = (
        reward_gap[positive_gap].mean()
        if positive_gap.any()
        else metric_zero
    )
    squared_weight_sum = gate_weights.square().sum()
    safe_squared_weight_sum = torch.where(
        squared_weight_sum > 0,
        squared_weight_sum,
        torch.ones_like(squared_weight_sum),
    )
    effective_sample_size = weight_sum.square() / safe_squared_weight_sum
    metrics = {
        "sasrec_teacher_better_rate": (teacher_rank < student_rank).float().mean(),
        "sasrec_distilled_rate": distill_mask.float().mean(),
        "sasrec_student_target_rank": _masked_mean(student_rank, all_examples),
        "sasrec_teacher_target_rank": _masked_mean(teacher_rank, all_examples),
        "sasrec_student_target_prob": _masked_mean(
            student_target_probability,
            all_examples,
        ),
        "sasrec_teacher_target_prob": _masked_mean(
            teacher_target_probability,
            all_examples,
        ),
        "sasrec_candidate_count": _masked_mean(
            candidate_mask.sum(dim=1),
            all_examples,
        ),
        "sasrec_student_reward": student_reward.mean(),
        "sasrec_teacher_reward": teacher_reward.mean(),
        "sasrec_reward_gap": reward_gap.mean(),
        "sasrec_gate_weight_mean": gate_weights.mean(),
        "sasrec_gate_nonzero_rate": distill_mask.float().mean(),
        "sasrec_gate_weight_p50": torch.quantile(gate_weights, 0.5),
        "sasrec_gate_weight_p90": torch.quantile(gate_weights, 0.9),
        "sasrec_positive_reward_gap_mean": positive_gap_mean,
        "sasrec_gate_effective_sample_size": effective_sample_size,
    }
    if teacher_reward_mode == "normalized_log_rank":
        metrics["sasrec_reward_available_item_count"] = (
            available_item_count.mean()
        )
    elif teacher_reward_mode == "candidate_bpr":
        metrics.update({
            "sasrec_student_bpr_loss": student_bpr_loss.mean(),
            "sasrec_teacher_bpr_loss": teacher_bpr_loss.mean(),
            "sasrec_bpr_negative_count": negative_count.float().mean(),
        })
    return loss, metrics


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train SASRec on an Recommendation DatasetDict."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
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
    valid_rows, valid_cold_items = build_eval_rows(
        expert_validation,
        train_data.n_items,
        train_item_mask=train_data.train_item_mask,
    )
    print(
        f"Train-only sequences: users={train_data.n_users} "
        f"items={train_data.n_items} "
        f"observed_items={int(train_data.train_item_mask.sum())} "
        f"examples={len(train_data.examples)}"
    )
    print(
        f"Train-derived validation rows: evaluated={len(valid_rows)} "
        f"cold_items={valid_cold_items}"
    )
    device = _resolve_device(args.device)
    model, training_info = train_sasrec(
        train_data=train_data,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
        seed=args.seed,
        valid_rows=valid_rows,
        valid_k=args.valid_k,
        eval_every=args.eval_every,
        patience=args.patience,
        eval_batch_size=args.eval_batch_size,
    )
    training_info = dict(training_info)
    training_info.update(
        {
            "dataset_dir": str(Path(args.dataset_dir)),
            "seed": args.seed,
            "validation_source": "train",
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.seed,
            "validation_rows": len(expert_validation),
            "fit_rows": len(fit_dataset["train"]),
            "valid_cold_items": valid_cold_items,
            "train_observed_items": int(train_data.train_item_mask.sum()),
            "train_examples": len(train_data.examples),
            "train_users": train_data.n_users,
        }
    )
    save_sasrec_checkpoint(args.output_path, model, train_data, training_info)
    print(f"Saved SASRec checkpoint to {args.output_path}")

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
        print(
            f"[{split_name}] hr@{args.valid_k}="
            f"{metrics[f'hr@{args.valid_k}']:.6f} "
            f"ndcg@{args.valid_k}={metrics[f'ndcg@{args.valid_k}']:.6f} "
            f"rows={int(metrics['eval_rows'])} cold_items={cold_items}"
        )


if __name__ == "__main__":
    main()
