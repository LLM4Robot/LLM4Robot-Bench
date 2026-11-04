# R5 control revalidation branch

`experiment/r5-control-revalidation` is a local experimental branch based on
PR #1 (`3fe766aae0a9bfb65902606a637234392181095d`). GPT task runs through the
normal CLI, batch/reproduction/fleet chain and explicit bottle experiment use
this profile by default. `main` is unchanged. This branch has not been pushed,
merged, deployed, or validated by new GPU/model runs.

Task files, assets, prompts, success criteria, cameras and original initial
layouts remain frozen. `tools/release/check_task_contract.py` must pass without
refreshing the lock. These runs are new control experiments, not replays of the
paper's scored trials. Selecting 270° remains an explicit initial-state choice.

## Control changes

- Enable the existing robot gravity compensation option in the effective run
  configuration. Keep the native Cartesian completion tolerance (5 mm for
  L3-01), joint limits and orientation limits.
- Solve Cartesian targets to 0.1 mm and require endpoint FK error below 0.25 mm
  (originally 1 mm and 2 mm). Keep the reviewed planner mirror and its source
  hash guard; do not silently fall back to the old planner on an import error.
- On a finger-only coarse `start_collision`, permit a narrowly scoped fallback
  for one Cartesian arm, at most 3 cm / 15.01°. Finger components use closed
  convex hulls; **each world mesh uses a solid convex enclosure**. The latter
  replaces the PR's unsigned BVH surface query, which can miss a finger fully
  contained in a closed mesh. Intersections and containment in either direction
  are rejected. Invalid/degenerate geometry fails closed.
- The world enclosure can fill concavities and join disconnected components.
  It can therefore reject clear narrow paths. This is an intentional
  conservative tradeoff, only in the fallback; simulation meshes and ordinary
  native collision checks are unchanged. Other-link, self, other-arm and payload
  checks remain in force. Candidate paths need a 25 µm clearance and bounded
  adaptive subdivision; unverified paths are rejected.
- On `goal_ik_not_found`, retain the PR's at most four numerical IK seeds and
  200 function evaluations per seed, then native whole-path checks and a final
  original-target check. The three-second pre-seed check is **not** a hard
  deadline inside native planning. The planner client and process supervisor
  remain responsible for their respective timeouts.
- Report measured residuals for all arms, including retained working-arm
  targets. Above the 0.5 mm reporting marker, native `completed` feedback is
  annotated `partial_execution`; native errors remain visible. This neither
  stops an episode nor selects recovery actions. No object truth, contact
  forces or evaluator flags are added to the actor observation.

Tighter planning and advisory feedback may increase rejection, runtime or model
calls. Compensation changes contact behavior; small planned FK errors do not
establish precise physical execution. No claim of universal improvement is made.

## CPU-only checks

With the project installed and its pinned assets available:

```bash
python tools/control_revalidation/run.py check --task L3-01
python tools/control_revalidation/run.py check \
  --experiment-config configs/experiments/l3_01_preloosen_270.json
python -m unittest discover -s tests -p 'test_control_revalidation*.py' -v
python tools/release/check_task_contract.py
python tools/verify_runtime_assets.py
```

The first check uses the default **180°** task. The second uses the existing
**270°** experiment and verifies the scene/scoring pose agreement, approximately
6.75 mm cap lift, 150 completed generations and 18,000 seconds. Both are offline
and do not call a model. Geometry regressions require the dependencies in
`tools/control_revalidation/requirements.txt`; without them those tests skip.
Run them with those dependencies before claiming the containment fix verified.

## Runtime and launch integration

Install `tools/control_revalidation/requirements.txt` into an appropriate
**cuRobo** environment, compatible with its Python/NumPy. No dependencies are
installed automatically. This profile currently supports a Linux simulator and
planner host; native Windows/WSL split deployment is not validated.

The profile reuses `GPT_BENCHMARK_PLANNER_PYTHON` or the configured planner Python
from `configs/runtime/planner.local.toml`. An explicit `R5_PRECISION_PYTHON` can
select a separate validation environment. Set it to the real cuRobo interpreter,
not `planner_python.py`. The scoped adapter restores the original environment
and policy registration on return or exception. Remote preflight checks the
planner dependencies before accepting a job.

Use the same Mac Codex → SSH bridge → remote supervisor chain for 270°:

```bash
python skills/r5-remote-deploy/scripts/fleet.py start --machine cluster \
  --tasks L3-01 \
  --experiment-config configs/experiments/l3_01_preloosen_270.json
```

Only run after the branch has been explicitly deployed and the selected GPU is
idle. This command calls the model and starts simulation. The branch changes do
not create credentials, change model transport, add retries or generate actions
outside the configured policy.

For a directly configured Linux host, an explicit general task run is:

```bash
python tools/control_revalidation/run.py run --task L2-01 --gpu 0 \
  --output runs/control_revalidation_001 \
  --model-config configs/models/gpt.local.toml
```

`run` keeps the framework supervisor alive, invokes the configured Isaac Python
as a worker, and forwards a monotonic deadline covering startup and model waits.
It reuses `robot_tasks.batch.run_job` for owned-process-tree cancellation and
cleanup. The extra 300 seconds is cleanup grace, not extra task time. Supplying
`--experiment-config` delegates to the existing bottle launcher rather than
reimplementing the 270° setup. Existing batch/fleet paths retain their supervision.

## Evidence and rollback

Each real episode has effective control metadata in `effective_config.json`,
plus a sidecar under the output root's `control-profiles/`. This saves the profile,
source hashes, angle (when explicitly selected), budgets and inherited deadline
before simulator startup. The native recorder still captures actions, model
requests/replies, measured feedback, scores and videos. Private geometric audit
logs remain outside model inputs. Batch resume now fingerprints the experimental
worker files as well as `src/`, refusing changed control code mid-batch.

The new CPU geometry regressions cover fully contained fingers, contained
obstacles, surface intersections, transformed obstacles, an intersecting path
with clear endpoints, conservative concavity rejection and invalid geometry.
They are synthetic checks, not robot trajectory or GPU performance evidence.
PR #1 describes earlier upstream GPU/physical diagnostics; those results were
not independently repeated by this branch's offline validation.

The existing 270° asset diagnostic permitted direct upward pull-off. This
control profile does not establish a required screw angle or repair that physical
constraint. Fresh GPU evaluation is still needed to quantify planning latency,
false rejections, physical tracking and task outcomes.

After saving local work, switching to `main` restores the released launch path.
Do not merge this branch or push/deploy it as an approved release until its
separate evaluation has been reviewed. Preserve all historical runs and scores.
