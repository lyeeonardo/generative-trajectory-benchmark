"""Validated access to the canonical paper episode results.

Paper-facing episode summaries should read the consolidated CSV through this
module instead of selecting individual experiment roots independently.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PAPER_RESULTS = Path("outputs/paper_ready/main/paper_raw_results.csv")
APPROVED_INCLUSION_RULE = "approved_non_smoke_per_episode_source"

SOURCE_EXPERIMENT_GROUPS: Mapping[str, str] = {
    "id_n_sweep_main": "id_n_sweep",
    "heldout_start_goal_main": "heldout_start_goal",
    "hidden_tilt_learned_followup": "hidden_tilt",
    "hidden_tilt_cem_reference": "hidden_tilt",
    "ood_gate_baseline_followup": "ood_gate",
    "ood_gate_oracle_followup": "ood_gate",
    "ood_gate_tuned_followup": "ood_gate",
    "equal_wall_clock_cpu": "equal_wall_clock_cpu",
}

EXPERIMENT_SOURCES: Mapping[str, tuple[str, ...]] = {
    "id_n_sweep": ("id_n_sweep_main",),
    "heldout_start_goal": ("heldout_start_goal_main",),
    "hidden_tilt": ("hidden_tilt_learned_followup", "hidden_tilt_cem_reference"),
    "ood_gate": (
        "ood_gate_baseline_followup",
        "ood_gate_oracle_followup",
        "ood_gate_tuned_followup",
    ),
    "equal_wall_clock_cpu": ("equal_wall_clock_cpu",),
}

REQUIRED_COLUMNS = frozenset(
    {
        "experiment",
        "variant",
        "model",
        "K",
        "candidate_budget",
        "scene_id",
        "seed",
        "status",
        "success",
        "paper_source_label",
        "paper_experiment_group",
        "paper_source_path",
        "paper_row_index_in_source",
        "paper_inclusion_rule",
    }
)


class PaperResultsError(ValueError):
    """Raised when the canonical paper result contract is violated."""


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise PaperResultsError(f"Expected boolean value, received {value!r}")


def as_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def as_int(value: Any) -> int:
    number = as_float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise PaperResultsError(f"Expected integer-like value, received {value!r}")
    return int(number)


def _semantic_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return (
        row.get("paper_source_label", ""),
        row.get("experiment", ""),
        row.get("variant", ""),
        row.get("model", ""),
        row.get("K", ""),
        row.get("scene_id", ""),
        row.get("seed", ""),
        row.get("equal_wall_clock_budget_ms", ""),
    )


def _validate_rows(
    rows: list[dict[str, str]],
    fieldnames: Iterable[str],
    *,
    require_all_sources: bool,
    require_existing_source_paths: bool,
) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    if missing_columns:
        raise PaperResultsError(f"Canonical paper CSV is missing columns: {missing_columns}")
    if not rows:
        raise PaperResultsError("Canonical paper CSV contains no rows")

    labels = {row["paper_source_label"] for row in rows}
    unknown_labels = sorted(labels.difference(SOURCE_EXPERIMENT_GROUPS))
    if unknown_labels:
        raise PaperResultsError(f"Unknown paper source labels: {unknown_labels}")
    if require_all_sources:
        missing_labels = sorted(set(SOURCE_EXPERIMENT_GROUPS).difference(labels))
        if missing_labels:
            raise PaperResultsError(f"Missing canonical paper source labels: {missing_labels}")

    bad_status = Counter(row.get("status", "") for row in rows if row.get("status", "") != "ok")
    if bad_status:
        raise PaperResultsError(f"Non-ok rows present in canonical results: {dict(bad_status)}")
    bad_inclusion = Counter(
        row.get("paper_inclusion_rule", "")
        for row in rows
        if row.get("paper_inclusion_rule", "") != APPROVED_INCLUSION_RULE
    )
    if bad_inclusion:
        raise PaperResultsError(f"Unapproved inclusion rules present: {dict(bad_inclusion)}")

    provenance_keys = [
        (row["paper_source_label"], row["paper_row_index_in_source"])
        for row in rows
    ]
    duplicate_provenance = len(provenance_keys) - len(set(provenance_keys))
    if duplicate_provenance:
        raise PaperResultsError(f"Duplicate provenance keys: {duplicate_provenance}")

    semantic_keys = [_semantic_key(row) for row in rows]
    duplicate_semantic = len(semantic_keys) - len(set(semantic_keys))
    if duplicate_semantic:
        raise PaperResultsError(f"Duplicate semantic episode keys: {duplicate_semantic}")

    for index, row in enumerate(rows, start=1):
        label = row["paper_source_label"]
        expected_group = SOURCE_EXPERIMENT_GROUPS[label]
        if row["paper_experiment_group"] != expected_group:
            raise PaperResultsError(
                f"Row {index} source {label!r} has paper_experiment_group "
                f"{row['paper_experiment_group']!r}; expected {expected_group!r}"
            )
        if row.get("K", "") and row.get("candidate_budget", ""):
            if as_int(row["K"]) != as_int(row["candidate_budget"]):
                raise PaperResultsError(
                    f"Row {index} has K={row['K']!r} but candidate_budget={row['candidate_budget']!r}"
                )
        as_bool(row["success"])
        if require_existing_source_paths and not Path(row["paper_source_path"]).is_file():
            raise PaperResultsError(
                f"Row {index} source path does not exist: {row['paper_source_path']}"
            )


@dataclass(frozen=True)
class PaperResults:
    """Immutable, validated view of the consolidated episode rows."""

    path: Path
    rows: tuple[dict[str, str], ...]
    sha256: str

    @property
    def source_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row["paper_source_label"] for row in self.rows).items()))

    def select(self, **filters: Any) -> list[dict[str, str]]:
        """Return rows matching exact values or a collection of accepted values."""

        selected: list[dict[str, str]] = []
        for row in self.rows:
            keep = True
            for key, expected in filters.items():
                if isinstance(expected, (set, frozenset, tuple, list)):
                    keep = row.get(key, "") in {str(value) for value in expected}
                else:
                    keep = row.get(key, "") == str(expected)
                if not keep:
                    break
            if keep:
                selected.append(dict(row))
        return selected

    def experiment_rows(self, experiment_group: str) -> list[dict[str, str]]:
        if experiment_group not in EXPERIMENT_SOURCES:
            raise KeyError(f"Unknown experiment group: {experiment_group}")
        return self.select(paper_source_label=EXPERIMENT_SOURCES[experiment_group])


def load_paper_results(
    path: str | Path = DEFAULT_PAPER_RESULTS,
    *,
    require_all_sources: bool = True,
    require_existing_source_paths: bool = False,
) -> PaperResults:
    """Load and validate the canonical consolidated episode CSV."""

    result_path = Path(path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Canonical paper result CSV not found: {result_path}")
    raw = result_path.read_bytes()
    with result_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    _validate_rows(
        rows,
        fieldnames,
        require_all_sources=require_all_sources,
        require_existing_source_paths=require_existing_source_paths,
    )
    return PaperResults(
        path=result_path,
        rows=tuple(rows),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
