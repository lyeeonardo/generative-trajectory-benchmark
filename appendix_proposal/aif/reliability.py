"""Smooth generator reliability gating for Experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def bounded_omega(logit: float, omega_min: float, omega_max: float) -> float:
    return float(omega_min + (omega_max - omega_min) * _sigmoid(logit))


@dataclass
class FixedReliability:
    """Backward-compatible fixed omega scaffold."""

    omega_t: float = 0.0

    def update(self, diagnostics: dict[str, object] | None = None) -> float:
        del diagnostics
        return float(self.omega_t)


@dataclass
class ReliabilityGate:
    mode: str = "smooth_gate"
    omega_min: float = 0.0
    omega_max: float = 1.0
    lambda_d: float = 0.35
    lambda_omega: float = 0.35
    b_omega: float = 3.0
    kappa_omega: float = 3.0
    w_AB: float = 1.0
    w_R: float = 0.0
    w_r: float = 0.5
    w_ood: float = 0.0
    w_q: float = 0.0
    w_inv: float = 0.0
    c_d: float = 0.5
    delayed_steps: int = 2
    seed: int | None = None
    ell_t: float | None = None
    dbar_t: float = 0.0
    omega_t: float = 1.0
    step_index: int = 0
    history: list[dict[str, float | str | bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = str(self.mode)
        self.ell_t = float(self.b_omega if self.ell_t is None else self.ell_t)
        self.omega_t = self._mode_initial_omega()
        self._rng = np.random.default_rng(self.seed)

    def _mode_initial_omega(self) -> float:
        if self.mode == "fixed_omega_0":
            return float(self.omega_min)
        if self.mode in {"fixed_omega_1", "overconfident_gate"}:
            return float(self.omega_max)
        return bounded_omega(float(self.ell_t), float(self.omega_min), float(self.omega_max))

    def _weighted_mismatch(self, signals: dict[str, Any]) -> float:
        z_ab = float(signals.get("z_delta_AB", signals.get("delta_AB", signals.get("mismatch_AB", 0.0))))
        z_r = float(signals.get("z_delta_R", signals.get("delta_R", signals.get("mismatch_R", 0.0))))
        kl_d = float(signals.get("Delta_d", signals.get("belief_kl", 0.0)))
        q_cand = float(signals.get("candidate_quality", signals.get("q_cand", 0.0)))
        invalidity = float(signals.get("candidate_invalidity", signals.get("p_invalid", 0.0)))
        residual = max(0.0, z_ab - float(self.c_d) * kl_d)
        ood = 1.0 if bool(signals.get("ood", signals.get("OOD", False))) else 0.0
        return float(
            self.w_AB * z_ab
            + self.w_R * z_r
            + self.w_r * residual
            + self.w_ood * ood
            + self.w_q * q_cand
            + self.w_inv * invalidity
        )

    def update(self, signals: dict[str, Any] | None = None) -> float:
        """Update omega_t from mismatch signals through a bounded smooth logit."""

        payload = dict(signals or {})
        mode = self.mode
        if mode == "fixed_omega_1":
            self.omega_t = float(self.omega_max)
        elif mode == "fixed_omega_0":
            self.omega_t = float(self.omega_min)
        elif mode == "random_gate":
            self.omega_t = float(self._rng.uniform(self.omega_min, self.omega_max))
        elif mode == "oracle_gate":
            self.omega_t = float(self.omega_min if bool(payload.get("ood", False)) else self.omega_max)
        elif mode == "delayed_gate":
            if self.step_index < int(self.delayed_steps):
                self.omega_t = float(self.omega_max)
            else:
                self._smooth_update(payload)
        elif mode == "overconfident_gate":
            self.omega_t = float(self.omega_max)
            self.dbar_t = 0.0
        elif mode == "smooth_gate":
            self._smooth_update(payload)
        else:
            raise ValueError(f"Unknown reliability gate mode {mode!r}.")
        self.history.append(
            {
                "step": float(self.step_index),
                "mode": mode,
                "omega_t": float(self.omega_t),
                "dbar_t": float(self.dbar_t),
                "ell_t": float(self.ell_t),
                "mismatch_t": float(self._weighted_mismatch(payload)),
                "z_delta_AB": float(payload.get("z_delta_AB", payload.get("delta_AB", 0.0))),
                "candidate_invalidity": float(payload.get("candidate_invalidity", 0.0)),
                "candidate_quality": float(payload.get("candidate_quality", 0.0)),
                "Delta_d": float(payload.get("Delta_d", payload.get("belief_kl", 0.0))),
                "ood": bool(payload.get("ood", False)),
            }
        )
        self.step_index += 1
        return float(self.omega_t)

    def _smooth_update(self, payload: dict[str, Any]) -> None:
        mismatch = self._weighted_mismatch(payload)
        self.dbar_t = float((1.0 - self.lambda_d) * self.dbar_t + self.lambda_d * mismatch)
        target_logit = float(self.b_omega - self.kappa_omega * self.dbar_t)
        self.ell_t = float((1.0 - self.lambda_omega) * float(self.ell_t) + self.lambda_omega * target_logit)
        self.omega_t = bounded_omega(float(self.ell_t), float(self.omega_min), float(self.omega_max))

    def rho_t(self) -> float:
        denom = max(float(self.omega_max - self.omega_min), 1e-12)
        return float(np.clip((float(self.omega_t) - float(self.omega_min)) / denom, 0.0, 1.0))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "omega_t": float(self.omega_t),
            "rho_t": self.rho_t(),
            "dbar_t": float(self.dbar_t),
            "ell_t": float(self.ell_t),
            "history": list(self.history),
        }


def make_reliability_gate(config: dict[str, Any] | None = None) -> ReliabilityGate:
    cfg = dict(config or {})
    return ReliabilityGate(
        mode=str(cfg.get("reliability_mode", cfg.get("gate_mode", "smooth_gate"))),
        omega_min=float(cfg.get("omega_min", 0.0)),
        omega_max=float(cfg.get("omega_max", 1.0)),
        lambda_d=float(cfg.get("lambda_d", 0.35)),
        lambda_omega=float(cfg.get("lambda_omega", 0.35)),
        b_omega=float(cfg.get("b_omega", 3.0)),
        kappa_omega=float(cfg.get("kappa_omega", 3.0)),
        w_AB=float(cfg.get("w_AB", 1.0)),
        w_R=float(cfg.get("w_R", 0.0)),
        w_r=float(cfg.get("w_r", 0.5)),
        w_ood=float(cfg.get("w_ood", 0.0)),
        w_q=float(cfg.get("w_q", 0.0)),
        w_inv=float(cfg.get("w_inv", 0.0)),
        c_d=float(cfg.get("c_d", 0.5)),
        delayed_steps=int(cfg.get("delayed_steps", 2)),
        seed=cfg.get("seed"),
    )
