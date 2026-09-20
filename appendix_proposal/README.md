# Appendix A: proposal models with simulator-based planning

This directory restores the exact early experiment used in Appendix A of the final paper. It is intentionally isolated from the main world-model code because its scientific question and data are different.

Here, BC-MDN, CVAE, diffusion, flow matching, and autoregressive models propose eight-step action sequences. The planner then uses **exact MuJoCo transitions** to roll out and score every candidate. The results establish that learned proposal distributions can improve candidate feasibility and finite-budget control, including under equal CPU planning targets and new start-goal layouts. They do not establish learned dynamics, world-model accuracy, or a complete GenAIF controller.

## Included scope

- Nominal six-scene candidate-budget sweep with three seeds.
- New start-goal six-scene transfer sweep with three seeds.
- Equal-wall-clock CPU calibration and evaluation.
- Separate proposal-training dataset.
- The five exact checkpoints referenced by the final experiment configs.
- Compact aggregate and per-episode results needed for Figure A1 and Tables A1–A2.

Hidden-tilt, OOD, horizon-ablation, video, and exploratory campaign artifacts from the early repository were excluded because they do not appear in the final paper.

## Run the appendix experiments

From the repository root, use the top-level environment:

```bash
cd appendix_proposal
sha256sum -c CHECKSUMS.sha256
uv run --project .. python scripts/run_paper_experiments.py --profile full --experiments id_n_sweep,heldout_start_goal --require_checkpoints
uv run --project .. python scripts/run_equal_wall_clock_cpu.py --mode all --torch-threads 1
```

The benchmark command writes new runs under `appendix_proposal/outputs/paper/`. The CPU command writes under `appendix_proposal/outputs/equal_wall_clock_cpu/`. To retrain the proposal models instead of using the included checkpoints:

```bash
uv run --project .. python scripts/train_paper_generators.py --device cuda
```

From the repository root, regenerate the final four-panel appendix figure with:

```bash
cd ..
uv run python scripts/build_appendix_figure.py
```

## Provenance

The bundle was recovered from `legacy_paper_experiment_2026-08-15.tar.gz`, SHA-256 `63285b750abf57bd1e48b52ad29226bd9a72cd90a296195e45aa8bbcecfaf96e`. Exact retained checkpoint hashes are listed in `CHECKSUMS.sha256`.
