# L3-01 additional preloosen experiment

`configs/experiments/l3_01_preloosen_270.json` selects the original L3-01 task with 270° initial preloosen, physical GPU 0, 150 completed model calls, and an 18,000-second wall-time budget. The supervisor includes simulator startup and model waiting in that budget. It allows an additional 300 seconds for cancellation, flushing and process cleanup; that grace is not extra task-execution time.

## Check without running

```bash
git submodule update --init assets
python tools/experiments/bottle_preloosen.py check --experiment-config configs/experiments/l3_01_preloosen_270.json
```

The check verifies the initial pose, active arms, prompt, budget and scene/scoring agreement. It does not start Isaac Sim or request a model response.

On `experiment/r5-control-revalidation`, this check also describes the branch's
default [control profile](CONTROL_REVALIDATION.md). Install its planner-only
dependencies in the selected remote cuRobo environment before running. The
270° initial state, 150-call budget, five-hour deadline and Mac Codex bridge
remain in the existing supervised launch chain; the additional control changes
are recorded separately from the paper baseline.

## Run on your configured remote machine

Configure your own runtime and local Codex bridge as described in `skills/r5-remote-deploy/SKILL.md`, then run from your local controller:

```bash
python skills/r5-remote-deploy/scripts/fleet.py start --machine cluster --tasks L3-01 --experiment-config configs/experiments/l3_01_preloosen_270.json
```

Machine credentials and host-specific configuration are not included. The launcher checks that the selected GPU is idle and preserves the benchmark's process supervisor.

## Interpretation

This is an additional initial-state experiment, not a replacement for the paper's scored trials. The scene and scoring use the same overridden initial pose. A prior external-load diagnostic at 270° allowed direct upward pull-off, so this setting does not establish an enforced screw-turn requirement. The default benchmark configuration, task prompt and success criteria remain separately reproducible.
