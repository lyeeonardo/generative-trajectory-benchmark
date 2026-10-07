# Benchmarking Generative Trajectory Models for Active-Inference Control

Official code and result artifacts for the paper by Yulin Li, Mohsen A. Jafari, and Andrea Matta.
Published as a conference paper at **IWAI 2026**.

[Read the paper on arXiv](https://arxiv.org/abs/2610.05692)

![Successful push–avoid–settle demonstrations at three board tilts](results/paper/media/environment.gif)

*The task in motion. Recorded reference-controller demonstrations at lateral tilts −15°, 0°, and +15° (left to right), all with a 20° uphill slope. In each panel, the pusher guides the ball around the obstacle and settles it inside the goal. The animation uses the same successful runs and camera views as the environment figure.*

The main paper is a **world-model benchmark**. Experiment 1 evaluates four shared trajectory-model families—Diffusion/DiT, autoregressive Transformer, joint CVAE, and flow matching—as action generators, controlled predictors, and observation-likelihood models. Experiment 2 compares the four pretrained models and a recovery-fine-tuned Diffusion variant on identical recorded actions and observations after an unannounced 0° → +15° tilt change. Only the recovery variant receives additional offline training; all weights remain fixed during evaluation.

Appendix A is a separate, earlier **proposal-model benchmark**. It uses exact MuJoCo transitions to score proposed action sequences. That is appropriate for studying proposal quality and finite-budget search, but it is not a learned-world-model experiment and does not demonstrate the complete GenAIF loop. Its code, separate dataset, checkpoints, and measurements are isolated in [appendix_proposal/](appendix_proposal/README.md).

## Results at a glance

![Figure 2: E1 action-generation, control, and controlled-prediction results](results/paper/figures/figure2_world_models.png)

*Figure 2. Experiment 1 compares action-generation/control performance under correct and wrong tilt labels (left) and controlled prediction error over horizons H=1–6 (right). Error bars and bands resample the frozen evaluation blocks; they do not represent retraining variation.*

![Experiment 2 live hidden-tilt belief update beside the matched MuJoCo replay](results/paper/media/e2_belief_demo.gif)

*Experiment 2 demo. The first ten transitions use 0° lateral tilt; the board then switches without announcement to +15°. The left panel updates the hidden-tilt posterior from each new observation while the right panel replays the identical recorded actions in MuJoCo. The visible board rotation is smoothed between observations 10 and 11 for presentation; the recorded experiment uses an instantaneous switch, and the posterior holds at observation 10 during the visual transition. This animation shows a representative original-model replay, not the recovery-fine-tuned model. Actions do not adapt to the belief. Figure 3 and Table 3 below compare all five evaluation arms over the same 12 switched trials.*

### E2: Belief adaptation and recovery experience

E2 replays 12 trajectories with a lateral-tilt change after action 10, alongside
12 matched no-switch controls. Starting from a uniform belief over −15°, 0°, and
+15°, each model predicts the observed transition under all three hypotheses and
updates their probabilities using the same fixed transition prior. Observation 11
is the first to reflect the switch. The filter receives neither the switch indicator
nor its timing, and uses no posterior reset or test-time refitting.

The original Diffusion model identifies the change fastest among the four pretrained
families, but later confidence is unstable while the recorded flat-board actions
continue. The recovery-fine-tuned variant reduces median identification delay from
three observations to one and raises correctness at observation 30 from 5/12 to
12/12. Its posterior stays at p(+15°) ≥ 0.9 throughout observations 16–30 in all
12 switched trials. Lines below show means; shaded bands are pointwise 95%
bootstrap intervals over source trajectories.

![E2 belief tracking with recovery-trained Diffusion added](results/paper/figures/figure3_belief_tracking_with_recovery.png)

| Model | Identification delay: median [min, max] | Identified by six | Correct at observation 30 |
| --- | ---: | ---: | ---: |
| Diffusion | 3 [2, 4] | 12/12 | 5/12 |
| Autoregressive | 5 [5, 8] | 10/12 | 7/12 |
| CVAE | 4 [3, 10] | 10/12 | 0/12 |
| Flow matching | 4 [3, 4] | 12/12 | 0/12 |
| Diffusion + recovery data | 1 [1, 2] | 12/12 | 12/12 |

This is the recovery-inclusive E2 comparison in the revised manuscript. The recovery
arm uses one reconstructed seed and its own calibration on previously inspected
trajectories; the comparison does not isolate recovery data as the sole cause of
improvement or establish performance on unseen cases. Every arm uses fixed action
replay, so E2 establishes belief-tracking capability, not successful ball control or
EFE-based action selection. See the [E2 protocol and results](docs/experiment2.md).

[Details and retained source data](results/paper/experiment2_recovery/README.md) ·
[PDF](results/paper/figures/figure3_belief_tracking_with_recovery.pdf) ·
[SVG](results/paper/figures/figure3_belief_tracking_with_recovery.svg) ·
[Reproduction code](scripts/build_e2_recovery_figure.py)

## Repository contents

- `aif/`, `data/`, `environment/`, `envs/`, `evaluation/`, `generators/`, `mujoco_task/`, `training/`: main-paper implementation.
- `configs/`: frozen E1/E2 protocols and the four seed-13 model recipes.
- `datasets/uphill_push_v1/`: main training data and cache.
- `datasets/evaluation/`: fixed development, calibration, and test banks.
- `results/paper/`: compact reports, source CSVs, paper tables, and regenerated figures.
- `appendix_proposal/`: self-contained archived proposal-only experiment.

The paper is hosted on [arXiv](https://arxiv.org/abs/2610.05692); a separate paper PDF
is not bundled in this repository. The original four-model E2 artifacts are retained
alongside the recovery-inclusive Figure 3 and Table 3.

Large intermediate checkpoints and rollout directories are intentionally excluded. The main E1 checkpoints can be regenerated from the frozen training contract; the five smaller Appendix A checkpoints used by the paper are included.

## Installation

Python 3.11 and MuJoCo are required. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Or use a standard virtual environment and install with `pip install -e '.[dev]'`.

## Verify the archived paper values

This command recomputes all five paper tables from the retained machine-readable artifacts, including the recovery row in Table 3 and the episode-weighted timing estimator used by Appendix Table A2:

```bash
uv run python scripts/verify_paper_results.py
```

Regenerate every paper figure from the retained sources:

```bash
uv run python scripts/build_benchmark_figures.py
uv run python scripts/experiment2_switching_plus15.py --output results/paper/experiment2 --plot-only
uv run python scripts/build_e2_recovery_figure.py
uv run python scripts/build_appendix_figure.py
uv run python scripts/render_environment_figure.py
```

Regenerate the three-tilt environment animation from its recorded successful states:

```bash
uv run python scripts/build_environment_gif.py
```

Regenerate the animated E2 demo by replaying its retained actions in MuJoCo; this performs no model inference:

```bash
uv run python scripts/build_e2_demo_gif.py
```

Regenerate the recovery-inclusive Figure 3 and Table 3 from the bundled
predictions, without training, model checkpoints, or access to the local experiments:

```bash
uv run python scripts/build_e2_recovery_figure.py
```

## Reproduce the main experiments

The complete E1 pipeline creates recipes, trains the four seed-13 models, selects checkpoints on development data, calibrates likelihoods, freezes the protocol, and runs the sealed test campaign:

```bash
uv run --extra dev python scripts/experiment1_full.py --stage all
uv run python scripts/build_benchmark_figures.py --recompute-source
```

The exact original run used an NVIDIA GPU with a 450 W power limit. Training is intentionally explicit and long-running. The original four-model E2 evaluation consumes the resulting selected checkpoints and frozen calibrations:

```bash
uv run python scripts/experiment2_switching_plus15.py --device cuda
```

The recovery arm's saved predictions, calibration metadata, and provenance are
retained in `results/paper/experiment2_recovery/`. Its figure-reproduction command
above does not retrain or freshly evaluate that checkpoint.

See [docs/reproducibility.md](docs/reproducibility.md) for artifact provenance, estimators, and the distinction between the main and appendix tracks.

## Tests

```bash
uv run --extra dev pytest
```

## License

Released under the [Apache License 2.0](LICENSE).

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff). Please cite the paper if you use this code or the released benchmark artifacts.
