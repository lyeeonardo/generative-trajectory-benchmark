"""Tilt-hypothesis rollout utilities for hidden-tilt AIF experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from aif.hidden_tilt_belief import HiddenTiltBelief
from aif.scoring import AIFScorer
from aif.tilt_context import snapshot_with_tilt


def evaluate_candidates_expected_tilt(
    *,
    candidates: np.ndarray,
    env_adapter,
    scorer: AIFScorer,
    belief,
    tilt_belief: HiddenTiltBelief,
    snapshot=None,
    mode: str = "exact_grid",
    m: int | None = None,
    seed: int | None = None,
    score_config: dict[str, Any] | None = None,
    epistemic_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Posterior-weighted candidate scores across tilt hypotheses.

    The live environment is restored before returning.
    """

    scores, probabilities, diagnostics = evaluate_candidates_tilt_distribution(
        candidates=candidates,
        env_adapter=env_adapter,
        scorer=scorer,
        belief=belief,
        tilt_belief=tilt_belief,
        snapshot=snapshot,
        mode=mode,
        m=m,
        seed=seed,
        score_config=score_config,
        epistemic_values=epistemic_values,
    )
    expected = scores @ probabilities
    return expected.astype(np.float32), diagnostics


def evaluate_candidates_tilt_distribution(
    *,
    candidates: np.ndarray,
    env_adapter,
    scorer: AIFScorer,
    belief,
    tilt_belief: HiddenTiltBelief,
    snapshot=None,
    mode: str = "exact_grid",
    m: int | None = None,
    seed: int | None = None,
    score_config: dict[str, Any] | None = None,
    epistemic_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return candidate cost under every selected tilt hypothesis.

    The matrix is shaped ``(candidate, hypothesis)``.  Keeping the complete
    distribution allows robust backups to optimize posterior mean, worst-case,
    or posterior CVaR without ever consulting the true tilt.
    """

    actions = np.asarray(candidates, dtype=np.float32)
    if actions.ndim != 3:
        raise ValueError("candidates must be shaped (K, H, action_dim).")
    live_snapshot = env_adapter.clone_state()
    base_snapshot = live_snapshot if snapshot is None else snapshot.copy()
    indices = tilt_belief.hypothesis_indices(mode=mode, m=m, seed=seed)
    probabilities = np.asarray(tilt_belief.probabilities, dtype=np.float64)
    selected_probs = probabilities[indices]
    selected_probs = selected_probs / max(float(np.sum(selected_probs)), 1e-12)
    score_matrix = np.zeros((actions.shape[0], indices.shape[0]), dtype=np.float64)
    epistemic = None if epistemic_values is None else np.asarray(epistemic_values, dtype=np.float32).reshape(-1)
    if epistemic is not None and epistemic.shape[0] != actions.shape[0]:
        raise ValueError("epistemic_values must match candidate count.")
    try:
        for column, hypothesis_index in enumerate(indices):
            tilted_snapshot = snapshot_with_tilt(base_snapshot, tilt_belief.tilt_grid[int(hypothesis_index)])
            for candidate_index, candidate in enumerate(actions):
                rollout = env_adapter.rollout_from_state(tilted_snapshot, candidate)
                if epistemic is not None:
                    rollout.raw_simulator_info["epistemic_value"] = float(epistemic[candidate_index])
                score_matrix[candidate_index, column] = float(
                    scorer.score_rollout(rollout, belief, config=score_config).G_total
                )
    finally:
        env_adapter.restore_state(live_snapshot)
    return score_matrix.astype(np.float32), selected_probs.astype(np.float32), {
        "mode": str(mode),
        "hypothesis_indices": [int(index) for index in indices.tolist()],
        "hypothesis_probabilities": selected_probs.astype(float).tolist(),
        "tilt_entropy": float(tilt_belief.entropy),
    }
