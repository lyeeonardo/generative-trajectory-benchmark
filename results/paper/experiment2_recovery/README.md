# Recovery-data augmentation for E2 belief tracking

![Original E2 curves with recovery-trained Diffusion](../figures/figure3_belief_tracking_with_recovery.png)

This figure is reproduced from saved measured predictions in the dissertation's
`chapter3/side_experiments/e2_recovery_retraining/figure_extension` experiment. The
compact inference archive and its original source contract are retained in `source/`.
The original four paper curves and confidence bounds are preserved unchanged.

The source experiment continued a reconstructed Diffusion model (seed 13) using
original data mixed with 10,080 unique physically measured recovery examples. It
added 1,048,576 sampled windows after 4,194,304 original-only base-training windows.
The continuation exposure comprised 786,432 native windows and 262,144 recovery
draws. These training counts describe the archived experiment; this figure rebuild
does not perform training or new model inference.

Everything needed to rebuild the figure and extended table is retained here or in
the original paper artifacts. The source experiment's local folder and checkpoints
are not needed for that rebuild. This package reproduces saved predictions, not
the original training process or a fresh checkpoint evaluation.

All five curves use the same 12 switched trajectories. Lateral tilt changes from
0° to +15° after the tenth action; observation 11 is the first affected posterior.
Recorded actions, observations, tilt hypotheses, and the belief-update rule are
preserved. The 12 unchanged companion trajectories are used only for alarm metrics.

| Measure | Recovery-trained Diffusion |
| --- | ---: |
| Identification delay: median [min, max] | 1 [1, 2] observations |
| Correct final tilt estimate | 12/12 trajectories |
| Identified the switch within six observations | 12/12 trajectories |
| Maintained p(+15°) ≥ 0.9 throughout observations 16–30 | 12/12 trajectories |
| Mean p(+15°) at observation 30 | 0.999592 |
| False tilt alarms on unchanged companion trajectories | 0/12 trajectories |

These measurements are recomputed from the archived per-trajectory posteriors in
`metrics.json`. The lowest individual posterior over observations 16–30 is 0.929758.
Identification means the first post-switch observation with p(+15°) ≥ 0.9;
an unchanged-companion alarm means any non-flat MAP tilt from observation 10 onward.
The delay is the crossing observation minus 10, matching Table 3: ten switched
trials cross at observation 11 and two at observation 12. The extended Table 3
retains all four original rows and adds the recovery-trained model.

This provides a working configuration for sustained belief tracking on these
previously inspected trajectories. The added curve uses one reconstructed training
seed and its own calibration. The comparison does not isolate recovery data as the
sole cause of improvement or establish performance on unseen cases. This figure
measures belief tracking under recorded actions and does not establish successful
ball control. Bootstrap intervals describe source-trajectory variation, not variation
across training seeds.

## Figure caption

Posterior probability of the +15° tilt before and after a change from 0° to +15° at
action 10. The orange curve adds Diffusion trained with recovery data to the four
original model curves. Lines show means over the same 12 switched trajectories;
shaded regions show 95% bootstrap intervals over source trajectories.

## Reproduce

From the repository root:

```bash
uv run python scripts/build_e2_recovery_figure.py
```

The script checks source hashes against the original inference contract, recomputes
likelihoods and posterior filtering, verifies the replay observations and original
four curves, and recomputes the fifth mean and all bootstrap intervals using 20,000
draws with seed 260917. It exports PNG, PDF, and SVG alongside the original figure,
plus the five-series curve CSV, recovery per-trial posteriors, metrics, and provenance.

- [PNG](../figures/figure3_belief_tracking_with_recovery.png)
- [PDF](../figures/figure3_belief_tracking_with_recovery.pdf)
- [SVG](../figures/figure3_belief_tracking_with_recovery.svg)
- [Curve values](belief_curve.csv)
- [Per-trial posterior values](recovery_trial_beliefs.csv)
- [Recomputed metrics](metrics.json)
- [Table 3 with recovery-trained Diffusion](../tables/table3_e2_with_recovery.csv)
- [Provenance](provenance.json)
