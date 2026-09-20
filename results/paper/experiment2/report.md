# Experiment 2: replayed hidden-tilt belief tracking

The first ten executed transitions use 0° lateral tilt; the next twenty use +15°. Observation 11 is the first posterior that incorporates a switched transition. Each of twelve held-out source episodes has a matched no-switch replay, and all models receive identical recorded actions and public observations.

Seed-13 weights, sampling settings, calibrated likelihoods, and the development-selected total hazard h=0.06 are frozen. There is no posterior reset, likelihood tempering, test-time refitting, action adaptation, or transition selection. All 24 trajectories contain 30 executed actions and no safety termination.

| Model | Delay median [min,max] | Identified by six | Correct at observation 30 | Maintained ≥0.9 over observations 16–30 |
|---|---:|---:|---:|---:|
| Diffusion/DiT | 3 [2,4] | 12/12 | 5/12 | 0/12 |
| Autoregressive Transformer | 5 [5,8] | 10/12 | 7/12 | 0/12 |
| Joint CVAE | 4 [3,10] | 10/12 | 0/12 | 0/12 |
| Flow Matching | 4 [3,4] | 12/12 | 0/12 | 0/12 |

Delay is the first post-switch observation at which p(+15°) reaches 0.9. Later declines occur even though +15° remains unchanged and every trajectory supplies all 30 observations. Thus rapid initial identification does not imply sustained belief tracking.

Confidence intervals resample the twelve source episodes 20,000 times with seed 260917. They do not include retraining variation. `belief_curve.csv` backs Figure 3; `trial_beliefs.csv` contains every model/trial/time posterior; `metrics.csv` contains the original aggregate export; likelihood arrays and replay trajectories are retained for audit.
