# Final paper artifacts

- `experiment1/`: compact completed E1 reports. Large checkpoints, raw predictive samples, and simulator rollouts are omitted.
- `experiment2/`: all 24 replay trajectories, per-model likelihood arrays, trial posteriors, aggregate curves, and report.
- `experiment2_recovery/`: exploratory recovery-trained Diffusion extension, including compact predictions and provenance; not part of the frozen paper results.
- `source/experiment1/`: fixed CSVs underlying the two-panel E1 figure.
- `source/environment/`: source images, measured trajectories, and provenance for the environment figure.
- `tables/`: values printed in Tables 1–3 and A1–A2; `table3_e2_with_recovery.csv` separately adds the exploratory recovery row.
- `figures/`: publication figures regenerated from the retained sources.
- `media/`: README animation and its machine-readable E2 case-selection record.

`scripts/verify_paper_results.py` checks the printed table values against the machine-readable reports rather than trusting the table CSVs themselves.

`scripts/build_e2_recovery_figure.py` independently checks the recovery extension's
bundled predictions, recomputes the posterior curves and metrics, and regenerates
its PNG/PDF/SVG figure and extended table without the disposable local experiments.
