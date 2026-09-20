# Retained dataset

`datasets/uphill_push_v1` is the only experiment dataset retained in the repository.

## Coverage

- 9 physical conditions: 3 lateral tilts × 3 obstacle positions
- 2 useful route sides per condition
- fixed start, goal, longitudinal tilt, and obstacle radius
- train, development, calibration, and test splits by parent rollout
- successful reference windows, imposed-action branches, terminal/waypoint completions, and recorded failures
- `H=6` action and next-observation targets

The exhaustive audit receipt is [`datasets/READINESS.json`](../datasets/READINESS.json), and the dataset contract is [`datasets/uphill_push_v1/contract.json`](../datasets/uphill_push_v1/contract.json).

## Limits

The splits reuse the same nine physical conditions, so they do not support an unseen-geometry claim. The reference set contains selected successes rather than an unbiased controller success rate. Recovery trajectories and failure examples are limited. These limits must remain visible in future E1/E2 reporting.

The active sampler uses the three tilt states and nine conditions while preserving parent-separated splits and TRAIN-only normalization; see [the fixes](diffusion_fixes.md). The recovered diffusion weights were accepted only after their embedded normalization, dataset manifest and task implementation matched the active artifacts.

## Active study audit

See [study_audit.md](study_audit.md) for split-specific event counts and frozen-bank support. The six-experiment plan requires derived sequence/constraint manifests and independent same-snapshot mode witnesses; it does not require replacing the retained base dataset.

## Settling coverage and full-fit evidence

Keep the existing five-step (0.25 s) slow/safe occupancy requirement. First audit actual training draws by approach, braking, in-goal moving and slow-goal phases, split by role, condition, source and requested/applied quality. Raw window count is not effective exposure. Preserve short terminal windows with valid masks; requiring six valid steps for every training draw would discard much of the relevant settling supervision.

TRAIN contains 2,320 slow in-goal windows, of which 605 have six valid future steps. The HIGH-labelled subset contains 244 slow in-goal windows, but only 13 full-H6 ones; reference trajectories contribute 580 slow windows but only 4 full-H6 ones, and reference quality labels are all NULL. These are overlapping windows, not independent braking/settling episodes. Terminal censoring explains much of the short horizon and is not a labelling defect.

Balance phase sampling and explicitly specify reference/quality conditioning before adding data. If this does not teach reliable braking/recovery, collect controlled near-goal position/velocity perturbations, unsuccessful attempts and short continuations after the success event. Continuation collection must have a separate contract: keep the original success event and do not pretend post-terminal actions belong to ordinary episodes. Preserve parent split boundaries and distinguish “reached” from “settled” success. The completed diffusion fit sampled 53,703 of 53,705 retained TRAIN windows, all conditions and all four phases without fallback. It nevertheless remained weak on broad closed-loop control, so future data expansion should target long off-reference recovery and braking continuations rather than simply repeat the same windows.

## Confirmed branch-context gap and bounded ablation

Every TRAIN branch target starts from the same five-observation/four-action history as a successful reference window at that parent and anchor: 17,050/17,050 safe intentional branches and 23,406/23,406 imposed-action branches match exactly. The branch futures are valuable dynamics data, but the original inputs provide no measured proposal context after a branch has moved off the reference path. See [`audit/branch_context_coverage.json`](audit/branch_context_coverage.json).
