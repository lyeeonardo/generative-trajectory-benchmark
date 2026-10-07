# Paper artifacts

- `experiment1/`: compact completed E1 reports. Large checkpoints, raw predictive samples, and simulator rollouts are omitted.
- `experiment2/`: all 24 replay trajectories, per-model likelihood arrays, trial posteriors, aggregate curves, and report.
- `experiment2_recovery/`: recovery-trained Diffusion arm in the revised manuscript's E2 comparison, including compact predictions and provenance.
- `source/experiment1/`: fixed CSVs underlying the two-panel E1 figure.
- `source/environment/`: source images, measured trajectories, and provenance for the environment figure.
- `tables/`: values for Tables 1–3 and A1–A2; `table3_e2_with_recovery.csv` is the revised five-row Table 3, while `table3_e2.csv` retains the original four rows.
- `figures/`: publication figures regenerated from retained sources; `figure3_belief_tracking_with_recovery.*` is the revised five-curve Figure 3.
- `media/`: README animation and its machine-readable E2 case-selection record.

`scripts/verify_paper_results.py` checks the table values, including the recovery
row, against the machine-readable reports rather than trusting the table CSVs themselves.

`scripts/build_e2_recovery_figure.py` independently checks the recovery arm's
bundled predictions, recomputes the posterior curves and metrics, and regenerates
its PNG/PDF/SVG figure and five-row table without the disposable local experiments.

The paper is linked at [arXiv:2610.05692](https://arxiv.org/abs/2610.05692), rather
than bundled as a separate PDF. These E2 descriptions follow the recovery-inclusive
revised manuscript while retaining the earlier four-model artifacts unchanged.
