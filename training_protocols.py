"""Dependency-light helpers for shared OutcomeGRPO/OPSD training protocols."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_FALSE_RESUME_VALUES = {"", "0", "false", "none", "no", "off"}
_LATEST_RESUME_VALUES = {"1", "true", "latest", "yes", "on"}


def combine_valid_test_datasets(
        eval_dataset: Mapping[str, Any],
        concatenate_fn: Optional[Callable] = None,
):
    """Return one ``test`` split containing valid then test rows.

    ``interaction_id`` is split-local in the prepared datasets and is used as
    a dictionary key while generated reasoning is gathered.  It therefore has
    to be regenerated after concatenation to prevent valid/test collisions.
    """

    if not isinstance(eval_dataset, Mapping):
        raise TypeError("Combined evaluation expects a split mapping.")
    missing = {"valid", "test"}.difference(eval_dataset)
    if missing:
        raise KeyError(
            "Combined evaluation requires valid and test splits; missing "
            + ", ".join(sorted(missing))
        )

    if concatenate_fn is None:
        from datasets import concatenate_datasets

        concatenate_fn = concatenate_datasets

    valid_dataset = eval_dataset["valid"]
    test_dataset = eval_dataset["test"]
    combined_dataset = concatenate_fn([valid_dataset, test_dataset])
    if "interaction_id" in combined_dataset.column_names:
        combined_dataset = combined_dataset.remove_columns("interaction_id")
    combined_dataset = combined_dataset.add_column(
        "interaction_id",
        list(range(len(combined_dataset))),
    )
    sizes = {
        "original_valid_rows": len(valid_dataset),
        "original_test_rows": len(test_dataset),
        "combined_test_rows": len(combined_dataset),
    }
    return {"test": combined_dataset}, sizes


def last_prefixed_metrics(
        log_history: Sequence[Mapping[str, Any]],
        prefix: str,
):
    """Return the last Trainer history row containing a prefixed metric."""

    for record in reversed(log_history):
        if any(str(key).startswith(prefix) for key in record):
            return dict(record)
    return None


def _read_complete_checkpoint(checkpoint: Path):
    """Return checkpoint Trainer state, rejecting incomplete artifacts."""

    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        raise ValueError(
            "resume_from_checkpoint must name a checkpoint-* directory, not "
            "the run directory or final_checkpoint."
        )
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(
            f"Incomplete checkpoint (missing trainer_state.json): {checkpoint}"
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        global_step = int(state["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid trainer_state.json in checkpoint: {checkpoint}"
        ) from error
    path_step = int(match.group(1))
    if global_step != path_step:
        raise ValueError(
            "Checkpoint directory step and trainer_state global_step differ: "
            f"{path_step} != {global_step}."
        )
    return state


def resolve_resume_checkpoint(resume_from_checkpoint: Any, run_dir: Any):
    """Resolve false/latest/explicit resume input inside one run directory."""

    run_path = Path(run_dir).expanduser().resolve()
    if isinstance(resume_from_checkpoint, bool):
        request = "latest" if resume_from_checkpoint else ""
    elif resume_from_checkpoint is None:
        request = ""
    else:
        request = str(resume_from_checkpoint).strip()

    normalized_request = request.lower()
    if normalized_request in _FALSE_RESUME_VALUES:
        return False

    if normalized_request in _LATEST_RESUME_VALUES:
        if not run_path.is_dir():
            raise FileNotFoundError(
                f"Cannot resume because the run directory does not exist: "
                f"{run_path}"
            )
        complete = []
        for candidate in run_path.glob("checkpoint-*"):
            try:
                state = _read_complete_checkpoint(candidate)
            except ValueError:
                continue
            complete.append((int(state["global_step"]), candidate.resolve()))
        if not complete:
            raise FileNotFoundError(
                f"No complete checkpoint-* directory found in {run_path}."
            )
        return str(max(complete, key=lambda pair: pair[0])[1])

    checkpoint = Path(request).expanduser().resolve()
    if checkpoint.parent != run_path:
        raise ValueError(
            "resume_from_checkpoint must belong to this exact run directory: "
            f"{run_path}"
        )
    _read_complete_checkpoint(checkpoint)
    return str(checkpoint)


def resolve_epoch_adapter_checkpoint(
        run_dir: Any,
        epoch: Any,
        metric_prefix: Optional[str] = "eval_test_",
):
    """Return one complete PEFT checkpoint for an exact finished epoch.

    A fixed-epoch checkpoint is usable by the OPSD handoff only after its
    Trainer state, LoRA adapter files, and (by default) epoch evaluation have
    all been persisted.  In-progress or malformed checkpoint directories are
    ignored so a polling launcher cannot race the Trainer's writes.
    """

    run_path = Path(run_dir).expanduser().resolve()
    try:
        target_epoch = float(epoch)
    except (TypeError, ValueError) as error:
        raise ValueError("epoch must be a finite positive number.") from error
    if not math.isfinite(target_epoch) or target_epoch <= 0:
        raise ValueError("epoch must be a finite positive number.")
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    matches = []
    for candidate in run_path.glob("checkpoint-*"):
        try:
            state = _read_complete_checkpoint(candidate)
            checkpoint_epoch = float(state["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isclose(
                checkpoint_epoch,
                target_epoch,
                rel_tol=0.0,
                abs_tol=1e-6,
        ):
            continue

        adapter_config = candidate / "adapter_config.json"
        adapter_weights = (
            candidate / "adapter_model.safetensors",
            candidate / "adapter_model.bin",
        )
        if not adapter_config.is_file() or not any(
                path.is_file() and path.stat().st_size > 0
                for path in adapter_weights
        ):
            continue

        if metric_prefix is not None:
            checkpoint_step = int(state["global_step"])
            evaluation_complete = False
            log_history = state.get("log_history", [])
            if not isinstance(log_history, list):
                continue
            for record in log_history:
                if not isinstance(record, Mapping):
                    continue
                try:
                    record_step = int(record.get("step", -1))
                except (TypeError, ValueError):
                    continue
                if record_step == checkpoint_step and any(
                        str(key).startswith(metric_prefix) for key in record
                ):
                    evaluation_complete = True
                    break
            if not evaluation_complete:
                continue

        matches.append((int(state["global_step"]), candidate.resolve()))

    if not matches:
        raise FileNotFoundError(
            f"No complete epoch {target_epoch:g} adapter checkpoint found "
            f"in {run_path}."
        )
    if len(matches) != 1:
        paths = ", ".join(
            str(path) for _, path in sorted(matches, key=lambda pair: pair[0])
        )
        raise RuntimeError(
            f"Expected one epoch {target_epoch:g} checkpoint in {run_path}, "
            f"found {len(matches)}: {paths}"
        )
    return str(matches[0][1])


def checkpoint_needs_evaluation(checkpoint: Any, prefix="eval_test_"):
    """Whether an epoch checkpoint lacks evaluation at its own global step."""

    state = _read_complete_checkpoint(Path(checkpoint).expanduser().resolve())
    checkpoint_step = int(state["global_step"])
    for record in state.get("log_history", []):
        try:
            record_step = int(record.get("step", -1))
        except (TypeError, ValueError):
            continue
        if record_step == checkpoint_step and any(
                str(key).startswith(prefix) for key in record
        ):
            return False
    return True
