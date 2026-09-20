# Reproducibility map

The final paper contains two deliberately different tracks.

## Main paper: learned world models

Experiment 1 uses one joint trajectory model per family. The same model provides goal-conditioned action proposals, predictions under imposed action sequences, and likelihoods for hidden-tilt inference. The primary control setting is `K=1`, horizon `H=6`, with correct-tilt, wrong-tilt, and agreement-reuse arms. The fixed seed is 13. The complete protocol is in `configs/experiment1_full.json`; compact completed reports are in `results/paper/experiment1/`.

Experiment 2 is the replayed 0° → +15° switch study in `configs/experiment2_switching_plus15.json`. Twelve switched trajectories and twelve matched no-switch trajectories use the same recorded actions and public observations for every model. The posterior transition hazard is fixed at `0.06`. This is belief tracking under replay, not an action-selection experiment. Its trial-level posteriors and likelihood arrays are in `results/paper/experiment2/`.

## Appendix A: proposal models with simulator planning

Appendix A predates the joint world-model study. Learned models propose action sequences, but exact MuJoCo transitions propagate each candidate during planning. It therefore tests proposal quality, feasibility, finite-budget search, gating, and CPU planning tradeoffs. It does **not** test learned state transitions, learned dynamics, or the complete GenAIF loop.

The appendix uses a separate demonstration archive and different checkpoints. The complete retained bundle is in `appendix_proposal/`. The final scope is limited to the nominal candidate-budget sweep, new start-goal transfer, and equal-wall-clock CPU comparison. Hidden-tilt and OOD development studies from the early repository are not part of the final paper and were removed.

## Retained result estimators

- E1 success intervals resample the ten locked proposal RNG trials while retaining the fixed physical cases.
- E1 prediction curves resample 18 reference-trajectory blocks and pool valid endpoints.
- E2 intervals resample 12 source episodes; recovery is the first post-switch observation with `p(+15°) >= 0.9`.
- Appendix A1 pools five proposal families with equal family weight.
- Appendix Table A2 reports the equal-weight mean across the 18 episode-level `planning_time_per_decision` values. The retained `per_budget_summary.csv` additionally reports pooled per-step timings, which are a different estimator.

Run `uv run python scripts/verify_paper_results.py` to derive and validate the paper tables from these artifacts.
