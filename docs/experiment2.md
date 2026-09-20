# Experiment 2: replayed hidden-dynamics belief tracking

Experiment 2 evaluates sequential inference with the same four frozen seed-13 world models used in Experiment 1. It does not run a controller or let the models choose different actions.

Twelve held-out flat-board source trajectories are replayed in two matched conditions. The no-switch condition remains at 0° lateral tilt. The switched condition executes ten flat-board transitions and then rotates the complete physical state into a +15° board frame before action 11, without advancing time or changing the public history. The next twenty recorded actions are replayed unchanged. Every model therefore receives the same actions, public observations, and physical trajectories.

The hidden state is one of −15°, 0°, or +15°. A symmetric three-state transition law uses total hazard 0.06: probability 0.94 of remaining in the current state and 0.03 for each alternative. At every observation, each world model supplies a one-step likelihood under each tilt hypothesis, and Bayes filtering updates the categorical posterior. There is no switch indicator, posterior reset, tempering, test-time fitting, or action adaptation.

The primary adaptation metric is the number of post-switch observations until p(+15°) first reaches 0.9. The experiment also reports identification by six observations, final maximum-posterior correctness at observation 30, and whether confidence remains at least 0.9 throughout observations 16–30.

The frozen protocol is in `configs/experiment2_switching_plus15.json`, the runner is `scripts/experiment2_switching_plus15.py`, and all retained trial-level evidence is in `results/paper/experiment2/`.
