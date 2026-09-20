"""Uncertainty summaries for the paired paper evaluation design.

The default settings implement ``docs/statistical_analysis_contract.md``.
Only NumPy is required so these routines work in the repository's base
environment.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_811


def default_uncertainty_metadata() -> dict[str, Any]:
    """Return manifest-ready metadata for the paper's uncertainty contract."""

    return {
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "cluster_bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "bootstrap_random_seed": DEFAULT_BOOTSTRAP_SEED,
        "cluster_columns": ["scene_id", "seed"],
        "architecture_weighting": "equal_within_cluster",
        "training_variability_included": False,
    }


def campaign_uncertainty_metadata() -> dict[str, Any]:
    """Return the stronger multi-training-seed campaign inference contract."""

    return {
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "crossed_cluster_bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "bootstrap_random_seed": DEFAULT_BOOTSTRAP_SEED,
        "crossed_cluster_axes": ["training_seed", "scene_id"],
        "within_cell_reduction": "equal_mean_over_eval_seeds_and_fixed_architecture_panel",
        "training_variability_included": True,
        "paired_method_contrasts": True,
        "simultaneous_stratum_intervals": "max_t",
    }


@dataclass(frozen=True)
class IntervalEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    method: str
    n_observations: int
    n_clusters: int
    n_resamples: int | None = None
    random_seed: int | None = None
    n_left_observations: int | None = None
    n_right_observations: int | None = None
    n_training_clusters: int | None = None
    n_scene_clusters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    median: float
    q1: float
    q3: float
    p90: float
    minimum: float
    maximum: float
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_confidence_level(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return array[np.isfinite(array)]


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> IntervalEstimate:
    """Return a Wilson score interval for a binomial proportion."""

    _validate_confidence_level(confidence_level)
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return IntervalEstimate(
        estimate=proportion,
        ci_low=max(0.0, center - half),
        ci_high=min(1.0, center + half),
        confidence_level=confidence_level,
        method="wilson_score",
        n_observations=total,
        n_clusters=total,
    )


def distribution_summary(values: Iterable[float]) -> DistributionSummary:
    """Return robust descriptive statistics for finite observations."""

    array = _finite_values(values)
    if array.size == 0:
        raise ValueError("at least one finite value is required")
    return DistributionSummary(
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        q1=float(np.quantile(array, 0.25)),
        q3=float(np.quantile(array, 0.75)),
        p90=float(np.quantile(array, 0.90)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        n_observations=int(array.size),
    )


def _cluster_reductions(
    values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    *,
    within_cluster: Callable[[np.ndarray], float],
) -> tuple[list[Hashable], np.ndarray, int]:
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must have the same length")
    if len(values) == 0:
        raise ValueError("at least one observation is required")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    n_observations = 0
    for value, cluster_id in zip(values, cluster_ids):
        number = float(value)
        if not math.isfinite(number):
            continue
        grouped[cluster_id].append(number)
        n_observations += 1
    if not grouped:
        raise ValueError("at least one finite observation is required")
    keys = sorted(grouped, key=repr)
    reduced = np.asarray(
        [within_cluster(np.asarray(grouped[key], dtype=np.float64)) for key in keys],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(reduced)):
        raise ValueError("within_cluster produced a non-finite value")
    return keys, reduced, n_observations


def cluster_bootstrap_interval(
    values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    *,
    within_cluster: Callable[[np.ndarray], float] = np.mean,
    across_clusters: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IntervalEstimate:
    """Percentile cluster bootstrap with equal weight per resampling block.

    Repeated observations inside a cluster (for example, the fixed architecture
    panel) are reduced first. Clusters are then sampled with replacement and
    receive equal weight in the paper-level estimand.
    """

    _validate_confidence_level(confidence_level)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    keys, reduced, n_observations = _cluster_reductions(
        values,
        cluster_ids,
        within_cluster=within_cluster,
    )
    estimate = float(across_clusters(reduced))
    if not math.isfinite(estimate):
        raise ValueError("across_clusters produced a non-finite estimate")
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, len(keys), size=(n_resamples, len(keys)))
    bootstrap = np.asarray(
        [across_clusters(reduced[index]) for index in sample_indices],
        dtype=np.float64,
    )
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        method="percentile_cluster_bootstrap",
        n_observations=n_observations,
        n_clusters=len(keys),
        n_resamples=n_resamples,
        random_seed=random_seed,
    )


def paired_cluster_bootstrap_difference(
    left_values: Sequence[float],
    left_cluster_ids: Sequence[Hashable],
    right_values: Sequence[float],
    right_cluster_ids: Sequence[Hashable],
    *,
    within_cluster: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_identical_clusters: bool = True,
) -> IntervalEstimate:
    """Bootstrap the paired difference ``left - right`` over matched blocks."""

    _validate_confidence_level(confidence_level)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    left_keys, left_reduced, left_n = _cluster_reductions(
        left_values,
        left_cluster_ids,
        within_cluster=within_cluster,
    )
    right_keys, right_reduced, right_n = _cluster_reductions(
        right_values,
        right_cluster_ids,
        within_cluster=within_cluster,
    )
    left_map = dict(zip(left_keys, left_reduced))
    right_map = dict(zip(right_keys, right_reduced))
    left_set = set(left_map)
    right_set = set(right_map)
    if require_identical_clusters and left_set != right_set:
        missing_left = sorted(right_set.difference(left_set), key=repr)
        missing_right = sorted(left_set.difference(right_set), key=repr)
        raise ValueError(
            "Paired cluster sets differ; "
            f"missing from left={missing_left}, missing from right={missing_right}"
        )
    common = sorted(left_set.intersection(right_set), key=repr)
    if not common:
        raise ValueError("No paired clusters are shared")
    differences = np.asarray([left_map[key] - right_map[key] for key in common], dtype=np.float64)
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, len(common), size=(n_resamples, len(common)))
    bootstrap = np.mean(differences[sample_indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        method="paired_percentile_cluster_bootstrap",
        n_observations=min(left_n, right_n),
        n_clusters=len(common),
        n_resamples=n_resamples,
        random_seed=random_seed,
        n_left_observations=left_n,
        n_right_observations=right_n,
    )


def _crossed_matrix(
    values: Sequence[float],
    training_cluster_ids: Sequence[Hashable],
    scene_cluster_ids: Sequence[Hashable],
    *,
    within_cell: Callable[[np.ndarray], float],
    require_complete_grid: bool,
) -> tuple[list[Hashable], list[Hashable], np.ndarray, int]:
    if not (len(values) == len(training_cluster_ids) == len(scene_cluster_ids)):
        raise ValueError("values and crossed cluster IDs must have the same length")
    grouped: dict[tuple[Hashable, Hashable], list[float]] = defaultdict(list)
    n_observations = 0
    for value, training_id, scene_id in zip(values, training_cluster_ids, scene_cluster_ids):
        number = float(value)
        if math.isfinite(number):
            grouped[(training_id, scene_id)].append(number)
            n_observations += 1
    if not grouped:
        raise ValueError("at least one finite crossed-cluster observation is required")
    training_keys = sorted({key[0] for key in grouped}, key=repr)
    scene_keys = sorted({key[1] for key in grouped}, key=repr)
    expected = {(training_id, scene_id) for training_id in training_keys for scene_id in scene_keys}
    missing = expected.difference(grouped)
    if missing and require_complete_grid:
        raise ValueError(f"Crossed cluster grid is incomplete; missing {sorted(missing, key=repr)[:8]}")
    matrix = np.full((len(training_keys), len(scene_keys)), np.nan, dtype=np.float64)
    training_index = {key: index for index, key in enumerate(training_keys)}
    scene_index = {key: index for index, key in enumerate(scene_keys)}
    for (training_id, scene_id), cell in grouped.items():
        reduced = float(within_cell(np.asarray(cell, dtype=np.float64)))
        if not math.isfinite(reduced):
            raise ValueError("within_cell produced a non-finite crossed-cluster value")
        matrix[training_index[training_id], scene_index[scene_id]] = reduced
    return training_keys, scene_keys, matrix, n_observations


def _crossed_bootstrap_distribution(
    matrix: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    training_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_training, n_scenes = matrix.shape
    if training_indices is None:
        training_indices = rng.integers(0, n_training, size=(n_resamples, n_training))
    scene_indices = rng.integers(0, n_scenes, size=(n_resamples, n_scenes))
    bootstrap = np.empty((n_resamples,), dtype=np.float64)
    for index in range(n_resamples):
        sampled = matrix[np.ix_(training_indices[index], scene_indices[index])]
        bootstrap[index] = float(np.nanmean(sampled))
    return bootstrap, training_indices


def crossed_cluster_bootstrap_interval(
    values: Sequence[float],
    training_cluster_ids: Sequence[Hashable],
    scene_cluster_ids: Sequence[Hashable],
    *,
    within_cell: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_complete_grid: bool = True,
) -> IntervalEstimate:
    """Independently resample training-seed and scene axes of a crossed panel."""

    _validate_confidence_level(confidence_level)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    training_keys, scene_keys, matrix, n_observations = _crossed_matrix(
        values,
        training_cluster_ids,
        scene_cluster_ids,
        within_cell=within_cell,
        require_complete_grid=require_complete_grid,
    )
    estimate = float(np.nanmean(matrix))
    bootstrap, _ = _crossed_bootstrap_distribution(
        matrix,
        n_resamples=n_resamples,
        rng=np.random.default_rng(random_seed),
    )
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        method="percentile_crossed_cluster_bootstrap",
        n_observations=n_observations,
        n_clusters=len(training_keys) * len(scene_keys),
        n_resamples=n_resamples,
        random_seed=random_seed,
        n_training_clusters=len(training_keys),
        n_scene_clusters=len(scene_keys),
    )


def paired_crossed_cluster_bootstrap_difference(
    left_values: Sequence[float],
    left_training_cluster_ids: Sequence[Hashable],
    left_scene_cluster_ids: Sequence[Hashable],
    right_values: Sequence[float],
    right_training_cluster_ids: Sequence[Hashable],
    right_scene_cluster_ids: Sequence[Hashable],
    *,
    within_cell: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IntervalEstimate:
    """Paired left-minus-right contrast with crossed seed/scene resampling."""

    left_training, left_scenes, left_matrix, left_n = _crossed_matrix(
        left_values,
        left_training_cluster_ids,
        left_scene_cluster_ids,
        within_cell=within_cell,
        require_complete_grid=True,
    )
    right_training, right_scenes, right_matrix, right_n = _crossed_matrix(
        right_values,
        right_training_cluster_ids,
        right_scene_cluster_ids,
        within_cell=within_cell,
        require_complete_grid=True,
    )
    if left_training != right_training or left_scenes != right_scenes:
        raise ValueError("Paired crossed-cluster axes differ between methods")
    difference = left_matrix - right_matrix
    base = crossed_cluster_bootstrap_interval(
        difference.reshape(-1).tolist(),
        [training_id for training_id in left_training for _ in left_scenes],
        [scene_id for _ in left_training for scene_id in left_scenes],
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    return IntervalEstimate(
        **{
            **base.to_dict(),
            "method": "paired_percentile_crossed_cluster_bootstrap",
            "n_observations": min(left_n, right_n),
            "n_left_observations": left_n,
            "n_right_observations": right_n,
        }
    )


def paired_crossed_cluster_bootstrap_ratio(
    numerator_values: Sequence[float],
    numerator_training_cluster_ids: Sequence[Hashable],
    numerator_scene_cluster_ids: Sequence[Hashable],
    denominator_values: Sequence[float],
    denominator_training_cluster_ids: Sequence[Hashable],
    denominator_scene_cluster_ids: Sequence[Hashable],
    *,
    within_cell: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IntervalEstimate:
    """Paired ratio of means with shared crossed seed/scene resampling."""

    _validate_confidence_level(confidence_level)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    left_training, left_scenes, left_matrix, left_n = _crossed_matrix(
        numerator_values,
        numerator_training_cluster_ids,
        numerator_scene_cluster_ids,
        within_cell=within_cell,
        require_complete_grid=True,
    )
    right_training, right_scenes, right_matrix, right_n = _crossed_matrix(
        denominator_values,
        denominator_training_cluster_ids,
        denominator_scene_cluster_ids,
        within_cell=within_cell,
        require_complete_grid=True,
    )
    if left_training != right_training or left_scenes != right_scenes:
        raise ValueError("Paired crossed-cluster axes differ between methods")
    denominator = float(np.mean(right_matrix))
    if denominator <= 0.0:
        raise ValueError("Paired ratio denominator mean must be positive")
    estimate = float(np.mean(left_matrix) / denominator)
    rng = np.random.default_rng(random_seed)
    training_indices = rng.integers(0, len(left_training), size=(n_resamples, len(left_training)))
    scene_indices = rng.integers(0, len(left_scenes), size=(n_resamples, len(left_scenes)))
    bootstrap = np.empty((n_resamples,), dtype=np.float64)
    for index in range(n_resamples):
        selector = np.ix_(training_indices[index], scene_indices[index])
        sampled_denominator = float(np.mean(right_matrix[selector]))
        if sampled_denominator <= 0.0:
            raise ValueError("Paired ratio bootstrap denominator must be positive")
        bootstrap[index] = float(np.mean(left_matrix[selector]) / sampled_denominator)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        method="paired_percentile_crossed_cluster_bootstrap_ratio",
        n_observations=min(left_n, right_n),
        n_clusters=len(left_training) * len(left_scenes),
        n_resamples=n_resamples,
        random_seed=random_seed,
        n_left_observations=left_n,
        n_right_observations=right_n,
        n_training_clusters=len(left_training),
        n_scene_clusters=len(left_scenes),
    )


def simultaneous_crossed_cluster_intervals(
    contrasts: dict[str, tuple[Sequence[float], Sequence[Hashable], Sequence[Hashable]]],
    *,
    within_cell: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Max-t simultaneous intervals for a family of stratum contrasts."""

    _validate_confidence_level(confidence_level)
    if not contrasts:
        raise ValueError("At least one contrast is required")
    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, tuple[int, int, int]] = {}
    shared_training: list[Hashable] | None = None
    for label, (values, training_ids, scene_ids) in contrasts.items():
        training, scenes, matrix, n_observations = _crossed_matrix(
            values,
            training_ids,
            scene_ids,
            within_cell=within_cell,
            require_complete_grid=True,
        )
        if shared_training is None:
            shared_training = training
        elif training != shared_training:
            raise ValueError("Simultaneous contrasts must share the same training-seed clusters")
        matrices[str(label)] = matrix
        metadata[str(label)] = (n_observations, len(training), len(scenes))
    assert shared_training is not None
    rng = np.random.default_rng(random_seed)
    training_indices = rng.integers(0, len(shared_training), size=(n_resamples, len(shared_training)))
    estimates = {label: float(np.mean(matrix)) for label, matrix in matrices.items()}
    bootstrap = {
        label: _crossed_bootstrap_distribution(
            matrix,
            n_resamples=n_resamples,
            rng=rng,
            training_indices=training_indices,
        )[0]
        for label, matrix in matrices.items()
    }
    standard_errors = {label: max(float(np.std(values, ddof=1)), 1e-12) for label, values in bootstrap.items()}
    max_t = np.max(
        np.stack(
            [np.abs((bootstrap[label] - estimates[label]) / standard_errors[label]) for label in matrices],
            axis=1,
        ),
        axis=1,
    )
    critical = float(np.quantile(max_t, confidence_level))
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, Any] = {}
    for label in matrices:
        point_low, point_high = np.quantile(bootstrap[label], [alpha, 1.0 - alpha])
        n_observations, n_training, n_scenes = metadata[label]
        half = critical * standard_errors[label]
        intervals[label] = {
            "estimate": estimates[label],
            "pointwise_ci_low": float(point_low),
            "pointwise_ci_high": float(point_high),
            "simultaneous_ci_low": float(estimates[label] - half),
            "simultaneous_ci_high": float(estimates[label] + half),
            "bootstrap_standard_error": standard_errors[label],
            "n_observations": n_observations,
            "n_training_clusters": n_training,
            "n_scene_clusters": n_scenes,
        }
    return {
        "confidence_level": confidence_level,
        "method": "max_t_crossed_cluster_bootstrap",
        "family_size": len(intervals),
        "critical_value": critical,
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "intervals": intervals,
    }
