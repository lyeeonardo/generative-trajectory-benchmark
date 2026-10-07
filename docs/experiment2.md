# Experiment 2: belief adaptation and recovery experience

Experiment 2 evaluates sequential inference with the four pretrained seed-13 world
models used in Experiment 1 and a Diffusion variant fine-tuned offline on recovery
data. Only the recovery variant receives additional training; all model weights
remain fixed during evaluation. Every arm replays the same recorded actions and
observations. E2 does not run a controller or let the models choose different actions.

Twelve held-out flat-board source trajectories are replayed in two matched conditions. The no-switch condition remains at 0° lateral tilt. The switched condition executes ten flat-board transitions and then rotates the complete physical state into a +15° board frame before action 11, without advancing time or changing the public history. The next twenty recorded actions are replayed unchanged. Every model therefore receives the same actions, public observations, and physical trajectories.

The hidden state is one of −15°, 0°, or +15°. A symmetric three-state transition law uses total hazard 0.06: probability 0.94 of remaining in the current state and 0.03 for each alternative. At every observation, each world model supplies a one-step likelihood under each tilt hypothesis, and Bayes filtering updates the categorical posterior. There is no switch indicator, posterior reset, tempering, test-time fitting, or action adaptation.

The primary adaptation metric is the number of post-switch observations until p(+15°) first reaches 0.9. The experiment also reports identification by six observations, final maximum-posterior correctness at observation 30, and whether confidence remains at least 0.9 throughout observations 16–30.

## Recovery-data arm

The retained recovery run continued a reconstructed seed-13 Diffusion model using
original data mixed with 10,080 unique physically measured recovery examples. It
added 1,048,576 sampled training windows after 4,194,304 original-only base windows.
The 12 switched trajectories, recorded actions and observations, tilt hypotheses,
and belief-update rule are unchanged from the original comparison.

| Model | Identification delay: median [min, max] | Identified by six | Correct at observation 30 |
| --- | ---: | ---: | ---: |
| Diffusion | 3 [2, 4] | 12/12 | 5/12 |
| Autoregressive | 5 [5, 8] | 10/12 | 7/12 |
| CVAE | 4 [3, 10] | 10/12 | 0/12 |
| Flow matching | 4 [3, 4] | 12/12 | 0/12 |
| Diffusion + recovery data | 1 [1, 2] | 12/12 | 12/12 |

The recovery-trained model maintains p(+15°) ≥ 0.9 throughout observations 16–30
in all 12 switched trajectories, with mean p(+15°) = 0.999592 at observation 30
and no false tilt alarms on the 12 unchanged companions. The plotted means and
pointwise 95% bootstrap intervals use the same 12 source trajectories in every arm.

The original models' later confidence fluctuations occur while recorded actions
designed for a flat board continue after the change. Recovery experience targets
the resulting unfamiliar states and transitions. The recovery arm uses one
reconstructed training seed and its own calibration, so this comparison does not
isolate recovery data as the sole cause of improvement or establish performance
on unseen trajectories. It demonstrates inference under replay, not adaptive action
selection, information-seeking control, or successful ball control.

## Retained artifacts and reproduction

- `configs/experiment2_switching_plus15.json` and `scripts/experiment2_switching_plus15.py`: original four-model protocol and runner.
- `results/paper/experiment2/`: original replay trajectories, likelihoods, and trial posteriors, retained unchanged.
- `results/paper/experiment2_recovery/`: saved recovery predictions, trial posteriors, metrics, and provenance.
- `results/paper/figures/figure3_belief_tracking_with_recovery.*`: recovery-inclusive Figure 3.
- `results/paper/tables/table3_e2_with_recovery.csv`: recovery-inclusive Table 3; `table3_e2.csv` retains the original four rows.

Rebuild the five-curve figure and table without retraining or model inference:

```bash
uv run python scripts/build_e2_recovery_figure.py
uv run python scripts/verify_paper_results.py
```

Paper: [arXiv:2610.05692](https://arxiv.org/abs/2610.05692). This E2 description
follows the recovery-inclusive revised manuscript; the original four-model artifacts
remain available for comparison with the earlier version.
