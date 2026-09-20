# Uphill push–avoid–settle dataset v1

**Created and audited:** 2026-09-15. This is a new, separately collected dataset for the agreed controlled task. No previous training data or checkpoints were overwritten.

## Task

- Board: 1.6 × 2.0 m; ball radius 0.04 m.
- Fixed ball start `(0, -0.65)` and goal center `(0, 0.84)`; goal radius 0.12 m.
- Pusher begins in the same board-relative pose behind the ball in every condition.
- Longitudinal slope: +20° (uphill toward the goal).
- Lateral slopes: −15°, 0°, +15°.
- Obstacle centers: `(−0.20, 0)`, `(0, 0)`, `(+0.20, 0)`; radius 0.10 m.
- Nine conditions; both actual left and right routes are represented in every condition.
- Success: five consecutive 0.05 s steps inside the goal with ball speed <0.10 m/s.
- Failure: ball–obstacle collision, ball leaving the board, or 100-step timeout. First goal entry alone is not success. Rod–obstacle contact is not a separate termination event.

## Contents

216 successful reference episodes: 12 independently varied successful controller trajectories per condition and route. Their collection required **4,097 candidate reference attempts**, with all attempt parameters/outcomes retained. This is selected demonstration collection, not a 100% reference-policy success-rate claim. The earlier controller-development searches are retained separately under `experiments/task_design/dataset_development/`.

The reference uses a smooth timed route with feedback on ball position/velocity and gravity compensation. Controller parameters vary across episodes. The nominal start and goal do not vary. The MuJoCo dynamics are deterministic; different seeds produce different controller parameters/action plans, not independent sensor-noise realizations.

At every fourth reference step, thirteen bounded H6 action plans are executed from an exactly restored physical state: reference, slower/faster, mild perturbations, hold, reverse, stronger perturbations, random constant/sequence actions, and lateral perturbations. These measured branches provide successful, unsuccessful and off-reference transitions. They are short interventions; a dedicated long recovery-rollout collection is not included in this version.

| Final split | Parent reference episodes | Retained windows |
| --- | ---: | ---: |
| Train | 144 | 53,705 |
| Development | 18 | 6,522 |
| Calibration | 18 | 6,666 |
| Test | 36 | 13,212 |
| Total | 216 | 80,105 |

The original development collection contained two episodes per condition/route: ordinal 0 is development, ordinal 1 is calibration. This rule is applied before any model use and depends only on episode index. Every split contains all nine conditions. All descendants of a reference episode stay in that episode's final split. Exact duplicate context/action windows are removed; train takes precedence, then the original development pool, then test. Removed duplicates remain in raw source files.

These are held-out trajectories/controller-parameter draws within the same nine conditions, **not unseen-geometry generalization**. Windows within a trajectory are correlated. Do not treat 80,105 windows as independent experiments.

## Layout

| Location | Content |
| --- | --- |
| `contract.json` | Fixed task/collection parameters, reference seed policies, source hashes |
| `episodes/<original split>/<condition and side>/` | Actions, observations, events, full private replay states; episode metadata and all search attempts |
| `reference_summary.json` | Quotas and collection attempt counts for all 54 original groups |
| `windows/<original split>/` | Raw measured windows before exact deduplication and DEV/CAL subdivision |
| `cache/{train,development,calibration,test}/` | Final model arrays as `.npy` files |
| `cache/normalization.json` | Normalization fitted to retained TRAIN arrays only |
| `cache/hashes.json` | Hashes of every final cache artifact |
| `cache/assembly.json` | Final split counts, scenario coverage, sources and quality labels |
| `split_assignment.json` | Authoritative parent episode → final split mapping |
| `audit.json` | Full reference replay, sampled branch replay and array/split checks |
| `manifest.json` | Dataset inventory and artifact hashes |

## Array schema and semantics

Public model inputs: `history_observations` (5×7), `history_actions` (4×3), `history_mask` (5), `geometry` (5), `tilt` (2 radians), and `time_fraction` (step/100). History is right-aligned. The 7 observation channels are ball x/y, ball vx/vy, pusher x/y, and pusher yaw. Geometry is goal x/y, obstacle x/y, radius. Do not feed full replay states or episode IDs to a model.

Targets: `actions` (6×3), `future_observations` (6×7), `valid` (6), `events` and `event_known` (6×5). Events are success, collision, fall, timeout, contact. Invalid padding is excluded from loss and does not provide completion targets. `source=1` marks successful reference windows; `source=0` marks intentional action branches; `source=2` marks imposed exploratory branches. `proposal_eligible` excludes unsafe intentional branches and includes successful reference windows.

`quality` has H1/H3/H6 labels: NULL=0, LOW=1, HIGH=2. These are relative safe-branch cost tertiles within the same anchor, using the inherited local operational cost. They are **not episode-success probabilities** and do not directly encode a velocity/dwell penalty. Successful reference windows remain NULL. The actual settling requirement is represented by measured observations and success events. `returns` stores negative inherited local horizon cost.

`scenario_id=0..8`: obstacle index ×3 + lateral index, with obstacle order left/center/right and lateral order −15/0/+15. `tilt_id=0..2`. Other identifiers are provenance fields.

### Training integration

The numerical array layouts and normalization match the model encoders, but old samplers hardcode 1,200 scenarios and six tilts. **Use a sampler driven by these nine scenarios and three labels.** Do not run the old fixed-six-tilt data preparation or relabel old weights as fitted to this dataset. All models trained on this version must share its split mapping and TRAIN normalization.

## Verification performed

- Four focused tests pass for settling, high-speed goal entry, timeout, collision priority, and physical-state/settle-counter restoration.
- All 216 reference episodes, totaling 19,761 physical transitions, replay exactly (observations, events, settling counters).
- 2,808 branch sequences (all 13 branch families at one middle anchor per reference episode), totaling 16,659 transitions, replay exactly. This is a sampled branch audit, not full replay of every branch.
- Every retained array row checked for finite data, bounded actions, contiguous validity, no valid steps after a terminal event, correct condition labels, all-nine-condition coverage and split assignment.
- No retained exact context/action duplicates or shared parent episodes across splits.
- Normalization independently recomputed from TRAIN and matched exactly.

## Reproduction

From the repository root:

```bash
.venv/bin/python experiments/task_design/build_dataset.py --stage all --workers 6
.venv/bin/python experiments/task_design/tools/finalize.py
.venv/bin/python experiments/task_design/tools/render_dataset.py
.venv/bin/python -m pytest experiments/task_design/tests -q
```

Completed reference/branch units are hash-checked and reused. Reassembly precedes final DEV/CAL packaging. Use a fresh output for changed generation code/settings. The finalize and render tools currently target `datasets/uphill_v1` explicitly.

## Review GIFs

[Gallery](../../experiments/task_design/dataset_gifs/README.md). Three GIFs each show left/center/right obstacle positions at a fixed lateral slope, covering all nine settings. A fourth shows both routes in the same centered-obstacle condition. These are measured reference demonstrations, not trained-model results.
