# Opt-in R5 control revalidation

This profile packages the L3-01 control diagnosis without replacing the frozen
paper baseline. All files under `src/`, task configurations, asset locks,
prompts, scoring, camera layouts and initial object layouts remain unchanged.
Runs using this profile must be labelled separately from baseline runs.

## What changes

- Enable the existing `simulation.robot_gravity_compensation` robot-control
  option in the run's effective configuration. The original task file is not
  edited. Keep its native Cartesian completion tolerance unchanged (5 mm for
  L3-01); do not confuse this with precise arrival or task success.
- Use a separate planner worker with 0.1 mm position-solving tolerance and
  0.25 mm endpoint FK acceptance, instead of 1 mm and 2 mm respectively.
  Orientation limits and real joint limits are retained.
- On a single-arm Cartesian `start_collision` request of at most 3 cm and
  15.01 degrees, try conservative closed convex hulls for each connected
  component of the original finger collision meshes. This is a replacement
  for coarse finger spheres, not a collision exemption. Other links, the other
  arm, self-collision and payload checks remain. A candidate must pass an FCL
  distance check with a 25 micrometre gap and adaptive swept-path subdivision;
  budget exhaustion, nonfinite distance or an uncertified segment rejects it.
- On `goal_ik_not_found`, try at most four numerical IK seeds within the
  original joint limits, 200 function evaluations per seed and an elapsed-time
  check before each seed. A candidate must satisfy 0.1 mm / 0.01 rad, then pass
  native whole-path joint planning, collision, payload and time-law checks.
  The final path is independently checked against the original requested pose.
  The elapsed-time check does not impose a hard deadline on native planning.
- Report measured arrival residuals for **every arm**, including an unchanged
  working-arm target during a camera-arm move. A completed action outside the
  0.5 mm reporting marker is labelled `partial_execution`, not precise arrival.
  This marker never terminates the episode, and repeated partial feedback does
  not terminate it either. Native errors/rejections remain visible. No recovery
  action is selected or executed by the wrapper.

The previous experiment-only 0.5 mm / 2 mm residual and consecutive-partial
episode-stop rules were diagnostic controls, not requirements of the original
benchmark. They are deliberately not included here. Object truth, contact
forces and evaluator flags are not inputs to the arrival annotator. A blocked
gripper or a small TCP residual is not proof of holding or cap detachment.

## CPU-only check

From the repository root, with its pinned assets available:

```bash
python tools/control_revalidation/run.py check --task L3-01
python -m unittest discover -s tests -p test_control_revalidation.py -v
python tools/release/check_task_contract.py
python tools/verify_runtime_assets.py
```

These commands never start Isaac, call a provider or actuate a robot. The check
verifies that the original task instruction, control instructions and schema
remain identical. Use the existing asset installation procedure; no lock needs
to be refreshed for this profile.

## Explicit simulation run

Use the existing installed Isaac and pinned cuRobo environments. Install
`tools/control_revalidation/requirements.txt` into the **cuRobo** environment
(FCL must match that environment's Python/NumPy). The interpreter shim is Linux
only; it is not a newly validated Windows/WSL deployment.

```bash
export R5_PRECISION_PYTHON=/absolute/path/to/curobo-python
export CUDA_VISIBLE_DEVICES=0
/absolute/path/to/isaac-python tools/control_revalidation/run.py run \
  --task L3-01 --gpu 0 --output runs/control_revalidation_001 \
  --model-config configs/models/gpt.local.toml
```

`run` is explicit and **does call the configured model**. It uses the existing
provider/login configuration and task budgets; it does not create credentials,
add skills, demonstrations or automatic model retries. Any retry policy already
declared by the selected task remains the native runner's policy. The output
directory must be new, and an occupied GPU is rejected without terminating any
existing process. Use a compatible physical GPU index in both places above.

The effective configuration and `control-profile.json` disclose the opt-in
control changes. The native recorder retains requested actions, planner results,
measured robot feedback and native scoring. Planner-only geometry audits stay
in the planner cache; they are not passed to the actor.

Rollback is selecting the original launcher/configuration: no original sources
or task files are rewritten. The standalone `native_precision.py` mirror is
intentional: changing a frozen shared worker would invalidate historical task
contracts. A source SHA guard and AST regression check require review/rebase if
the upstream worker changes. The only mirrored algorithm changes are two
precision constants and two endpoint-residual diagnostic fields.

## Evidence and limits

The packaged GPU planner regression was run on an NVIDIA GeForce RTX 4090 D
with zero model calls and zero robot actuation. Two previously rejected short
requests passed the conservative finger-mesh check; a normal request remained
accepted. Intersecting finger obstacles, a fully contained obstacle, non-finger
collision and an unreachable target remained rejected, and original finger
proxies were restored after rejection. The old finite-IK failure produced a
native-path-checked endpoint error of about 0.00220 mm and 0.00382 rad.

Earlier physical diagnostics of the same control changes measured empty-hand
1/2/5 mm lifts as 1.057/2.061/5.061 mm, and a privileged, deterministic native
robot sequence detached the cap and held it over 30 mm above the neck. These
are earlier diagnostic evidence, not new physics runs performed by this PR's
CPU/GPU tests. A corrected-Harness zero-shot run also succeeded once; that is
not a success-rate estimate or proof of universal asset fidelity.

Millimetre-scale execution residuals, grasp slip and loss of progress on
release still occur. This profile improves planning and preserves feedback;
it does not certify precise physical tracking or reliable contact manipulation.
No weights are trained or changed: the training stage is not applicable.
