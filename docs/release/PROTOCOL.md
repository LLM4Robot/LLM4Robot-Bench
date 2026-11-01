# Evaluation protocol and preserved variants

The software contains 16 task families and 32 independently selectable settings.
A new public suffix, `_variant01`, names the previously registered `_v3` setting.
Resolution returns the same setting for either suffix. Tasks use the current
numbers listed in the [numbering map](TASK_NUMBERING.md); configuration names
and public media follow those numbers. Delivered asset directories, historical
run IDs and recorded snapshots retain their original identities. Selecting both
suffix aliases of one current task in the same batch is rejected as a duplicate.

## What remains fixed

The renumbering changes identity fields and filenames, preserving prompts,
task parameters, scene/texture/mesh bytes, physical
material, collision geometry, initial pose, camera calibration/appearance,
scoring behavior and controller settings. After comparing the full effective
values against the preceding revision, `configs/benchmark/task_contract.json`
pins all 32 configurations and their protected files under their current names.
Derived numeric fingerprints use 12 significant digits (magnitudes below 1e-12 treated as zero) to
avoid Python/NumPy last-bit differences; the protected source/configuration
files still require exact SHA-256 equality. This normalization is only in the
checker and never changes a runtime value or model prompt. `tools/release/check_task_contract.py` recomputes effective
robot, simulation, execution, layout, instruction and scoring fingerprints.
`assets/runtime.lock.json` pins the complete 325-file asset closure separately.

## Existing differences are part of the protocol

| Task group | Original / alternative relationship |
|---|---|
| L1 (5 tasks) | Original placement and fixed selected P03 placement. The stored selection indices are 3, 3, 2, 1, 1 in task-ID order. |
| L2-01–L2-06 | Original placement and fixed P01 placement; existing task rules are retained. |
| L2-07 / L3-01 / L3-02 | Separate delivered scene settings, with their existing prompts, cameras and implementations. |
| L3-03 | Original and alternative use the same prompt and require containment of both books. The alternative translates the aligned stack and both rectangles by -0.10 m along world X. |
| L3-04 | Separate delivered alternative scene. Existing success criterion requires all three blocks. |

L2-03 requires the source cup to be empty and at least **6 of 12** beads to be in
the receiving cup. It does not succeed merely because six beads have arrived
while others are still in the source cup.

From 2026-10-23, the L3-03 alternative matches the paper's two-book containment
criterion instead of its earlier lower-book-only rule. Both settings require a
full turn during transfer, continuous stacking and tabletop support, and no
grasping or lifting. Historical runs retain their recorded prompts and scoring
rules; this update does not rescore them. Current budgets are listed below.

L1, L2-07 and L3 retain gravity compensation disabled and Cartesian completion tolerance
0.005 m; L2-01–L2-06 retain gravity compensation enabled and tolerance 0.001 m. These are
existing settings, not newly recommended universal values.

## Budgets

All 32 original and alternative settings default to **150 model calls or
18,000 seconds (5 hours)**, whichever limit is reached first. These defaults
apply to direct task execution, `tools/reproduce.py`, batch execution, and the
remote launcher. The additional L3-01 270-degree configuration uses the same limits.

`configs/benchmark/experiments.json` lists the current defaults, aligned with the
report on 2026-11-01. This describes new runs; existing run snapshots, historical
results and resumed batches retain their recorded budgets.

| Settings | Effective call cap | Wall-time cap | Call accounting |
|---|---:|---:|---|
| L1 originals / variants | 150 | 18,000 s | all requests |
| L2-01–L2-06 originals / variants | 150 | 18,000 s | all requests |
| L2-07 / L3 originals / variants | 150 | 18,000 s | completed generations |

An explicitly configured zero cap means disabled. `tools/reproduce.py` passes
each selected configuration's limits explicitly. CLI budget/model/seed overrides
must be reported as distinct conditions. Batch and remote execution include
simulator startup and model waiting in the wall-time deadline; the supervisor's
additional 300 seconds are for cancellation and cleanup, not task execution.

## Policy and execution

Model input comprises the task instruction, shared operating instructions,
configured camera images, robot state and execution feedback. It does not
receive evaluator object-state tables as a direct privileged input. The motion
planner and evaluator do use simulator geometry. Task-specific scoring and
execution safeguards are part of the system; do not describe it as just a raw
text prompt with no supporting machinery.

Within an episode, configured request history is available. Repeating a task
does not automatically add learned task skills or append past run archives to
model context. Real-robot zero-shot versus skill/history-enabled conditions must
be labeled independently; they are not interchangeable experimental populations.

## Reporting

Use the stored instruction, full policy instructions, effective configuration,
seed, model/transport/effort, planner revision, asset hash, calls, elapsed time,
actions, outcome and termination reason for each attempt. Keep all attempted
runs in the denominator and identify excluded infrastructure failures explicitly.
Do not combine different scoring rules, variants or budgets in a single success
rate without stratification. Videos are supporting evidence, not the sole source
of truth for the evaluator.

`batch_run.py --summarize PATH` writes JSON/CSV/Markdown and separates succeeded,
failed, invalid, timeout, interruption, infrastructure error, skipped and dry-run
outcomes. The release does not substitute invented totals for missing raw runs.
Raw experiment logs and real-robot materials are distributed separately when a
paper results release is finalized.
