"""Standalone LightGCN training and OPSD collaborative-rank utilities.

The core LightGCN propagation and BPR training code is adapted from the local
BigSmallRec project (``bigsmallrec/small_models/lightgcn.py``).  Recommendation-specific
code builds a train-only graph directly from its Hugging Face DatasetDict,
keeps item id 0 as padding, stores the string-user mapping in the checkpoint,
and exposes frozen final embeddings for optional OPSD rank distillation.

Example::

    python lightgcn.py \
        --dataset_dir data/CDs_and_Vinyl_0_2022-10-2023-10 \
        --output_path checkpoints/lightgcn_cds.pt \
        --device cuda:0
"""

from __future__ import annotations

import argparse
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


class LightGCN(nn.Module):
    """LightGCN with mean aggregation over layer-0 through layer-L."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        if n_users <= 0:
            raise ValueError("LightGCN requires at least one user.")
        if n_items <= 1:
            raise ValueError("LightGCN requires padding plus at least one real item.")
        if emb_dim <= 0 or n_layers < 0:
            raise ValueError("emb_dim must be positive and n_layers non-negative.")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    @staticmethod
    def _get_norm_adj(
        shifted_edge_index: torch.LongTensor,
        n_users: int,
        n_items: int,
    ) -> torch.Tensor:
        """Build D^-1/2 A D^-1/2 for the user-item bipartite graph."""

        n_nodes = n_users + n_items
        row = torch.cat([shifted_edge_index[0], shifted_edge_index[1]])
        col = torch.cat([shifted_edge_index[1], shifted_edge_index[0]])
        values = torch.ones(row.shape[0], device=row.device)

        degree = torch.zeros(n_nodes, device=row.device)
        degree.scatter_add_(0, row, values)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]

        return torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=norm_values,
            size=(n_nodes, n_nodes),
        ).coalesce()

    def forward(
        self,
        edge_index: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return graph-propagated ``(user_embeddings, item_embeddings)``."""

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        shifted = edge_index.clone()
        shifted[1] = shifted[1] + self.n_users
        adjacency = self._get_norm_adj(
            shifted,
            self.n_users,
            self.n_items,
        )
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )

        layer_embeddings = [embeddings]
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)

        final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss used by BigSmallRec LightGCN."""

    def forward(
        self,
        user_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        positive_scores = (user_emb * positive_emb).sum(dim=-1)
        negative_scores = (user_emb * negative_emb).sum(dim=-1)
        return -F.logsigmoid(positive_scores - negative_scores).mean()


@dataclass(frozen=True)
class RecommendationGraph:
    edge_index: torch.LongTensor
    user_id_to_index: dict[str, int]
    n_items: int
    train_item_mask: torch.BoolTensor

    @property
    def n_users(self) -> int:
        return len(self.user_id_to_index)


@dataclass(frozen=True)
class EvalRow:
    user_index: int
    history_item_ids: tuple[int, ...]
    target_item_id: int


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
        for name in ("user_id", history_column, target_column)
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
            "LightGCN requires dense item IDs aligned with Recommendation's item bank; "
            f"found {len(item_ids)} IDs spanning 0..{item_ids[-1]}."
        )
    return len(expected)


def build_train_graph(dataset_dict: Mapping[str, Any]) -> RecommendationGraph:
    """Build a deduplicated graph from the train split only.

    Every visible train history item and the train target are positive edges.
    Validation/test targets are never read by this function.
    """

    if "train" not in dataset_dict or "item_info" not in dataset_dict:
        raise KeyError("DatasetDict must contain train and item_info splits.")
    train_split = dataset_dict["train"]
    history_column, target_column = _split_columns(train_split)
    n_items = _validate_item_catalog(dataset_dict["item_info"])

    raw_user_ids = [str(user_id) for user_id in train_split["user_id"]]
    user_id_to_index = {
        user_id: index for index, user_id in enumerate(sorted(set(raw_user_ids)))
    }
    edge_pairs: set[tuple[int, int]] = set()
    histories = train_split[history_column]
    targets = train_split[target_column]
    for raw_user_id, history, target in zip(raw_user_ids, histories, targets):
        user_index = user_id_to_index[raw_user_id]
        positive_items = list(history) + [target]
        for raw_item_id in positive_items:
            item_id = int(raw_item_id)
            if item_id == PAD_ITEM_ID:
                continue
            if item_id < 0 or item_id >= n_items:
                raise ValueError(
                    f"Train item id {item_id} is outside catalog range 0..{n_items - 1}."
                )
            edge_pairs.add((user_index, item_id))

    if not edge_pairs:
        raise ValueError("No positive train interactions were found for LightGCN.")
    sorted_edges = sorted(edge_pairs)
    edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()
    train_item_mask = torch.zeros(n_items, dtype=torch.bool)
    train_item_mask[edge_index[1].unique()] = True
    train_item_mask[PAD_ITEM_ID] = False
    return RecommendationGraph(
        edge_index=edge_index,
        user_id_to_index=user_id_to_index,
        n_items=n_items,
        train_item_mask=train_item_mask,
    )


def build_eval_rows(
    split: Any,
    user_id_to_index: Mapping[str, int],
    n_items: int,
    train_item_mask: torch.BoolTensor | None = None,
) -> tuple[list[EvalRow], int, int]:
    """Convert one untouched evaluation split without adding graph edges."""

    history_column, target_column = _split_columns(split)
    if train_item_mask is not None:
        train_item_mask = train_item_mask.reshape(-1).bool().cpu()
        if train_item_mask.shape[0] != n_items:
            raise ValueError("train_item_mask does not match the item catalog.")
    rows: list[EvalRow] = []
    unknown_users = 0
    unavailable_items = 0
    for user_id, history, target in zip(
        split["user_id"],
        split[history_column],
        split[target_column],
    ):
        user_index = user_id_to_index.get(str(user_id))
        target_item_id = int(target)
        user_is_unknown = user_index is None
        item_is_unavailable = (
            target_item_id <= PAD_ITEM_ID
            or target_item_id >= n_items
            or (
                train_item_mask is not None
                and not bool(train_item_mask[target_item_id])
            )
        )
        if user_is_unknown:
            unknown_users += 1
        if item_is_unavailable:
            unavailable_items += 1
        if user_is_unknown or item_is_unavailable:
            continue
        history_item_ids = tuple(
            int(item_id)
            for item_id in history
            if PAD_ITEM_ID < int(item_id) < n_items
        )
        if target_item_id in history_item_ids:
            raise ValueError(
                f"Evaluation target {target_item_id} already occurs in its visible history."
            )
        rows.append(
            EvalRow(
                user_index=user_index,
                history_item_ids=history_item_ids,
                target_item_id=target_item_id,
            )
        )
    return rows, unknown_users, unavailable_items


def _build_user_pos_items(
    edge_index: torch.LongTensor,
) -> dict[int, set[int]]:
    user_pos_items: dict[int, set[int]] = {}
    for user_id, item_id in zip(
        edge_index[0].detach().cpu().tolist(),
        edge_index[1].detach().cpu().tolist(),
    ):
        user_pos_items.setdefault(user_id, set()).add(item_id)
    return user_pos_items


def _sample_negative_items(
    user_ids: torch.LongTensor,
    candidate_item_ids: np.ndarray,
    user_pos_items: Mapping[int, set[int]],
    device: torch.device,
    rng: np.random.Generator,
) -> torch.LongTensor:
    candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
    if candidate_item_ids.ndim != 1 or candidate_item_ids.size == 0:
        raise ValueError("Negative sampling requires train-observed item IDs.")
    if (candidate_item_ids <= PAD_ITEM_ID).any():
        raise ValueError("Padding cannot occur in the LightGCN negative pool.")
    negatives: list[int] = []
    for user_id in user_ids.detach().cpu().tolist():
        positives = user_pos_items.get(user_id, set())
        if len(positives) >= candidate_item_ids.size:
            raise ValueError(f"User index {user_id} has no available negative item.")
        while True:
            negative = int(
                candidate_item_ids[rng.integers(0, candidate_item_ids.size)]
            )
            if negative not in positives:
                negatives.append(negative)
                break
    return torch.tensor(negatives, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_lightgcn(
    model: LightGCN,
    edge_index: torch.LongTensor,
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

    was_training = model.training
    model.eval()
    user_embeddings, item_embeddings = model(edge_index)
    if candidate_item_mask is not None:
        candidate_item_mask = candidate_item_mask.reshape(-1).to(
            device=item_embeddings.device,
            dtype=torch.bool,
        )
        if candidate_item_mask.shape[0] != item_embeddings.shape[0]:
            raise ValueError("candidate_item_mask does not match LightGCN items.")
    hit_total = 0.0
    ndcg_total = 0.0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        user_indices = torch.tensor(
            [row.user_index for row in batch_rows],
            dtype=torch.long,
            device=item_embeddings.device,
        )
        scores = user_embeddings[user_indices] @ item_embeddings.t()
        if candidate_item_mask is None:
            scores[:, PAD_ITEM_ID] = -torch.inf
        else:
            scores[:, ~candidate_item_mask] = -torch.inf
        for row_index, row in enumerate(batch_rows):
            if row.history_item_ids:
                history = torch.tensor(
                    row.history_item_ids,
                    dtype=torch.long,
                    device=scores.device,
                )
                scores[row_index, history] = -torch.inf
        targets = torch.tensor(
            [row.target_item_id for row in batch_rows],
            dtype=torch.long,
            device=scores.device,
        )
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


def train_lightgcn(
    graph: RecommendationGraph,
    emb_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
    reg_weight: float = 1e-4,
    device: str | torch.device = "cuda",
    seed: int = 42,
    valid_rows: Sequence[EvalRow] | None = None,
    valid_k: int = 10,
    eval_every: int = 5,
    patience: int = 10,
) -> tuple[LightGCN, dict[str, Any]]:
    """Train the adapted BigSmallRec LightGCN with BPR."""

    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive.")
    if reg_weight < 0 or eval_every <= 0 or patience < 0:
        raise ValueError("reg_weight/patience must be non-negative; eval_every positive.")

    resolved_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    model = LightGCN(
        graph.n_users,
        graph.n_items,
        emb_dim=emb_dim,
        n_layers=n_layers,
    ).to(resolved_device)
    edge_index = graph.edge_index.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bpr_loss = BPRLoss()
    user_pos_items = _build_user_pos_items(graph.edge_index)
    negative_item_ids = torch.where(graph.train_item_mask)[0].cpu().numpy()
    n_interactions = edge_index.shape[1]

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -float("inf")
    best_epoch = 0
    evaluations_without_improvement = 0
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_interactions, device=resolved_device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_interactions, batch_size):
            indices = permutation[start : start + batch_size]
            users = edge_index[0, indices]
            positives = edge_index[1, indices]
            negatives = _sample_negative_items(
                users,
                negative_item_ids,
                user_pos_items,
                resolved_device,
                rng,
            )

            user_final, item_final = model(edge_index)
            loss = bpr_loss(
                user_final[users],
                item_final[positives],
                item_final[negatives],
            )
            if reg_weight > 0:
                regularization = (
                    model.user_embedding(users).pow(2).sum()
                    + model.item_embedding(positives).pow(2).sum()
                    + model.item_embedding(negatives).pow(2).sum()
                ) / (2 * users.shape[0])
                loss = loss + reg_weight * regularization

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            n_batches += 1

        final_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} - BPR Loss: {final_loss:.6f}")

        should_evaluate = bool(valid_rows) and (
            epoch % eval_every == 0 or epoch == epochs
        )
        if should_evaluate:
            metrics = evaluate_lightgcn(
                model,
                edge_index,
                valid_rows,
                k=valid_k,
                candidate_item_mask=graph.train_item_mask,
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
        "final_bpr_loss": final_loss,
    }


@torch.no_grad()
def save_lightgcn_checkpoint(
    output_path: str | Path,
    model: LightGCN,
    graph: RecommendationGraph,
    training_info: Mapping[str, Any] | None = None,
) -> None:
    """Save both trainable state and frozen final embeddings for OPSD."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    edge_index = graph.edge_index.to(device)
    was_training = model.training
    model.eval()
    user_embeddings, item_embeddings = model(edge_index)
    model.train(was_training)
    torch.save(
        {
            "format_version": 1,
            "source": "adapted from BigSmallRec small_models/lightgcn.py",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "edge_index": graph.edge_index.detach().cpu(),
            "user_id_to_index": dict(graph.user_id_to_index),
            "train_item_mask": graph.train_item_mask.detach().cpu(),
            "user_embeddings": user_embeddings.detach().float().cpu(),
            "item_embeddings": item_embeddings.detach().float().cpu(),
            "n_users": graph.n_users,
            "n_items": graph.n_items,
            "emb_dim": model.emb_dim,
            "n_layers": model.n_layers,
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
        raise TypeError("LightGCN checkpoint must contain a dictionary.")
    return checkpoint


@dataclass
class LightGCNEmbeddingBank:
    user_id_to_index: dict[str, int]
    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    train_item_mask: torch.BoolTensor
    source_path: str

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        expected_num_items: int | None = None,
    ) -> "LightGCNEmbeddingBank":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"LightGCN checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        required = {"user_id_to_index", "user_embeddings", "item_embeddings"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(
                "Checkpoint is missing Recommendation LightGCN artifacts "
                f"{sorted(missing)}. Train it with this repository's lightgcn.py."
            )
        user_embeddings = checkpoint["user_embeddings"].detach().float()
        item_embeddings = checkpoint["item_embeddings"].detach().float()
        if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ValueError("LightGCN final embeddings must be rank-2 tensors.")
        if user_embeddings.shape[1] != item_embeddings.shape[1]:
            raise ValueError("LightGCN user/item embedding dimensions differ.")
        if expected_num_items is not None and item_embeddings.shape[0] != expected_num_items:
            raise ValueError(
                "LightGCN/Recommendation item catalog mismatch: "
                f"checkpoint={item_embeddings.shape[0]}, Recommendation={expected_num_items}."
            )
        mapping = {
            str(user_id): int(index)
            for user_id, index in checkpoint["user_id_to_index"].items()
        }
        if sorted(mapping.values()) != list(range(user_embeddings.shape[0])):
            raise ValueError(
                "LightGCN user mapping must cover each embedding row exactly once."
            )
        train_item_mask = checkpoint.get("train_item_mask")
        if train_item_mask is None:
            edge_index = checkpoint.get("edge_index")
            if not isinstance(edge_index, torch.Tensor):
                raise KeyError(
                    "Checkpoint has neither train_item_mask nor edge_index."
                )
            train_item_mask = torch.zeros(
                item_embeddings.shape[0], dtype=torch.bool
            )
            train_item_mask[edge_index[1].long().unique()] = True
        train_item_mask = train_item_mask.detach().reshape(-1).bool()
        if train_item_mask.shape[0] != item_embeddings.shape[0]:
            raise ValueError("LightGCN train item mask has the wrong size.")
        if bool(train_item_mask[PAD_ITEM_ID]):
            raise ValueError("Padding item 0 cannot be train-observed.")
        if not train_item_mask.any():
            raise ValueError("LightGCN checkpoint has no train-observed items.")
        return cls(
            user_id_to_index=mapping,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            train_item_mask=train_item_mask,
            source_path=str(checkpoint_path),
        )

    def to(self, device: str | torch.device) -> "LightGCNEmbeddingBank":
        self.user_embeddings = self.user_embeddings.to(device)
        self.item_embeddings = self.item_embeddings.to(device)
        self.train_item_mask = self.train_item_mask.to(device)
        return self

    @torch.no_grad()
    def score_users(
        self,
        user_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Return full-catalog LightGCN dot scores and known-user mask."""

        indices = [self.user_id_to_index.get(str(user_id), -1) for user_id in user_ids]
        valid = torch.tensor(
            [index >= 0 for index in indices],
            dtype=torch.bool,
            device=self.item_embeddings.device,
        )
        safe_indices = torch.tensor(
            [max(index, 0) for index in indices],
            dtype=torch.long,
            device=self.item_embeddings.device,
        )
        user_embeddings = self.user_embeddings[safe_indices]
        scores = user_embeddings @ self.item_embeddings.t()
        if (~valid).any():
            scores[~valid] = 0
        return scores, valid


def _masked_mean(values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    if mask.any():
        return values[mask].float().mean()
    return values.float().sum() * 0.0


def lightgcn_rank_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    history_item_ids: torch.LongTensor,
    target_item_ids: torch.LongTensor,
    valid_user_mask: torch.BoolTensor,
    valid_item_mask: torch.BoolTensor | None = None,
    topk: int = 32,
    teacher_temperature: float = 0.2,
    student_temperature: float = 0.2,
    teacher_better_only: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distil LightGCN ranking on target/teacher-top-k/student-top-k union."""

    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError("Student/LightGCN scores must share shape [batch, items].")
    batch_size, n_items = student_scores.shape
    if n_items <= 1:
        raise ValueError("Collaborative rank distillation needs at least two items.")
    if history_item_ids.ndim != 2 or history_item_ids.shape[0] != batch_size:
        raise ValueError("history_item_ids must have shape [batch, history].")
    target_item_ids = target_item_ids.reshape(-1).to(student_scores.device)
    valid_user_mask = valid_user_mask.reshape(-1).to(student_scores.device)
    if target_item_ids.shape[0] != batch_size or valid_user_mask.shape[0] != batch_size:
        raise ValueError("Target/user masks do not match score batch size.")
    if topk < 0:
        raise ValueError("lightgcn_topk must be non-negative; use 0 for full catalog.")
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("LightGCN distillation temperatures must be positive.")
    if ((target_item_ids <= PAD_ITEM_ID) | (target_item_ids >= n_items)).any():
        raise ValueError("A LightGCN target item is outside the real item catalog.")

    history_item_ids = history_item_ids.to(student_scores.device)
    if valid_item_mask is None:
        valid_item_mask = torch.ones(
            n_items, dtype=torch.bool, device=student_scores.device
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
    available[:, PAD_ITEM_ID] = False
    valid_history = (
        (history_item_ids > PAD_ITEM_ID) & (history_item_ids < n_items)
    )
    if valid_history.any():
        rows = torch.arange(
            batch_size, device=student_scores.device
        ).unsqueeze(1).expand_as(history_item_ids)
        available[rows[valid_history], history_item_ids[valid_history]] = False
    if (~available.gather(1, target_item_ids.unsqueeze(1))).any():
        raise ValueError(
            "A LightGCN target is unavailable, padding, or already in history."
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

    distill_mask = valid_user_mask
    if teacher_better_only:
        distill_mask = distill_mask & (teacher_rank < student_rank)

    fill_value = -1e4
    student_logits = (student_float / student_temperature).masked_fill(
        ~candidate_mask, fill_value
    )
    teacher_logits = (teacher_float / teacher_temperature).masked_fill(
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
    if distill_mask.any():
        loss = per_example_kl[distill_mask].mean()
    else:
        loss = student_scores.sum() * 0.0

    student_target_probability = student_log_probs.gather(
        1, target_index
    ).exp().squeeze(1)
    teacher_target_probability = teacher_log_probs.gather(
        1, target_index
    ).exp().squeeze(1)
    valid_count = valid_user_mask.float().sum().clamp_min(1.0)
    metrics = {
        "lightgcn_mapped_user_rate": valid_user_mask.float().mean(),
        "lightgcn_teacher_better_rate": (
            (valid_user_mask & (teacher_rank < student_rank)).float().sum()
            / valid_count
        ),
        "lightgcn_distilled_rate": distill_mask.float().mean(),
        "lightgcn_student_target_rank": _masked_mean(
            student_rank, valid_user_mask
        ),
        "lightgcn_teacher_target_rank": _masked_mean(
            teacher_rank, valid_user_mask
        ),
        "lightgcn_student_target_prob": _masked_mean(
            student_target_probability, valid_user_mask
        ),
        "lightgcn_teacher_target_prob": _masked_mean(
            teacher_target_probability, valid_user_mask
        ),
        "lightgcn_candidate_count": _masked_mean(
            candidate_mask.sum(dim=1), valid_user_mask
        ),
    }
    return loss, metrics


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train BigSmallRec-style LightGCN on an Recommendation DatasetDict."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
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
    graph = build_train_graph(fit_dataset)
    valid_rows, valid_unknown, valid_cold_items = build_eval_rows(
        expert_validation,
        graph.user_id_to_index,
        graph.n_items,
        train_item_mask=graph.train_item_mask,
    )
    print(
        f"Train-only graph: users={graph.n_users} items={graph.n_items} "
        f"observed_items={int(graph.train_item_mask.sum())} "
        f"edges={graph.edge_index.shape[1]}"
    )
    print(
        f"Train-derived validation rows: evaluated={len(valid_rows)} "
        f"unknown_users={valid_unknown} cold_items={valid_cold_items}"
    )
    device = _resolve_device(args.device)
    model, training_info = train_lightgcn(
        graph=graph,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        reg_weight=args.reg_weight,
        device=device,
        seed=args.seed,
        valid_rows=valid_rows,
        valid_k=args.valid_k,
        eval_every=args.eval_every,
        patience=args.patience,
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
            "valid_unknown_users": valid_unknown,
            "valid_cold_items": valid_cold_items,
            "train_observed_items": int(graph.train_item_mask.sum()),
        }
    )
    save_lightgcn_checkpoint(args.output_path, model, graph, training_info)
    print(f"Saved LightGCN checkpoint to {args.output_path}")

    edge_index = graph.edge_index.to(device)
    for split_name in ("valid", "test"):
        rows, unknown_users, cold_items = build_eval_rows(
            dataset_dict[split_name],
            graph.user_id_to_index,
            graph.n_items,
            train_item_mask=graph.train_item_mask,
        )
        metrics = evaluate_lightgcn(
            model,
            edge_index,
            rows,
            k=args.valid_k,
            batch_size=args.eval_batch_size,
            candidate_item_mask=graph.train_item_mask,
        )
        print(
            f"[{split_name}] hr@{args.valid_k}={metrics[f'hr@{args.valid_k}']:.6f} "
            f"ndcg@{args.valid_k}={metrics[f'ndcg@{args.valid_k}']:.6f} "
            f"rows={int(metrics['eval_rows'])} unknown_users={unknown_users} "
            f"cold_items={cold_items}"
        )


if __name__ == "__main__":
    main()
