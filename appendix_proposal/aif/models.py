"""Named Stage 1 AIF model roles.

The MuJoCo rollout model is B_psi. The observation model A_phi is the
identity map in Stage 1 because the physical state is fully observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentityObservationModel:
    """Stage 1 A_phi: observation equals the known physical state vector."""

    def observe(self, s_t_phys: np.ndarray) -> np.ndarray:
        return np.asarray(s_t_phys, dtype=np.float32).copy()


@dataclass(frozen=True)
class CopiedSimulatorTransitionModel:
    """Stage 1 B_psi wrapper around ``env_adapter.rollout_from_state``."""

    env_adapter: object

    def rollout(self, state_snapshot, action_sequence: np.ndarray):
        return self.env_adapter.rollout_from_state(state_snapshot, action_sequence)
