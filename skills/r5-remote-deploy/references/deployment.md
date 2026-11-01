# Remote setup and operation

Run all controller commands from the repository root using a Python environment
with this project installed (`python -m pip install -e .`). Helpers require Linux
or WSL, SSH, and rsync on both ends. Isaac Sim and cuRobo remain separately
installed runtimes. See [local installation](../../../docs/release/INSTALL.md)
for the supported versions and task validation.

## Choose the deployment

Copy `skills/r5-remote-deploy/assets/fleet.example.json` to
`configs/runtime/remote.local.json`, then fill in:

- `host`, `user`, `machines`: SSH hostname/user and worker name-to-port mapping.
- `project`, `runtime`: absolute Linux paths for the checkout and its runtime.
- `isaac_root`, `planner_root`: existing Isaac installation and planner environment.
- `use_local_codex`: `false` for model access configured on the server; `true` for
  the local authenticated Codex bridge. Existing old configurations without this
  field retain their bridge behavior.
- `bridge_port`, `max_concurrent`: loopback forwarding port and local bridge
  request concurrency when that bridge is enabled. The server returns 429 when
  full, so select capacity to match the requested concurrency.
- Optional `guard_projects`: other checkout paths whose fleet-owned active jobs
  should block launching this worker. There are no hard-coded teammate projects.
- Optional `jump_host`, `direct_machines`: SSH jump host and workers that bypass it.

`GPT_BENCHMARK_FLEET_CONFIG=/path/to/config.json` selects another private file.
The old `tools/remote/fleet.local.json` is a fallback for existing deployments.
An explicit missing configuration is an error. The sample host is never contacted.
Credentials stay in SSH configuration or the model's own authentication store.

## Transfer and configure

Review the target path before sync: it mirrors project files and removes obsolete
files, while preserving paths in `assets/rsync-excludes.txt`. Pause changes to the
checkout during deployment. Do not start another job concurrently with a sync.

```bash
python skills/r5-remote-deploy/scripts/fleet.py sync --machine cluster
python skills/r5-remote-deploy/scripts/fleet.py bootstrap --machine cluster
python skills/r5-remote-deploy/scripts/fleet.py status --machine cluster
```

`sync` requires an idle target checkout and rsync. A missing remote rsync is an
installation issue, not an indication to disable exclusions or copy credentials.
`bootstrap` uses `tools/setup_runtime.sh`: it can install Debian rendering/video
libraries using the remote user's existing permissions, but does not install a
GPU driver or download Isaac. Where an administrator manages system packages,
run `tools/setup_runtime.sh --wrappers-only` on the host after setting the three
runtime variables documented in the local installation guide.

After transfer, use the remote framework Python to run:

```bash
python tools/verify_runtime_assets.py
python tools/release/check_task_contract.py
```

For exact transfer verification, prepare a JSON list of the transferred files
with `path`, `size`, and `sha256` and invoke
`scripts/verify_files.py /path/to/manifest.json` on the server. Neither check calls
a model. An initial-scene render is an optional separate GPU check; reserve an
idle GPU and follow `docs/release/INSTALL.md`.

## Model access

With `use_local_codex: false`, configure the model on the remote host as for a
local experiment. SSH forwarding and local Codex login are unnecessary.

With `use_local_codex: true`, install/authenticate Codex on the controller machine
and keep that machine online while experiments run:

```bash
python skills/r5-remote-deploy/scripts/fleet.py connect --machine cluster
```

This creates an authenticated loopback service and SSH reverse tunnel. Only its
random RPC token is copied to the selected runtime directory; account login
credentials stay local. The controller uses its current Python environment,
not a cached environment from a historical experiment. The configured runtime
path and port are explicitly propagated to every remote entry command.

`connect` checks login/connectivity without generating a model response.
`scripts/check_bridge.py` is a separate optional test that **does generate one
model response**; do not use it for routine status or synchronization checks.

## Run and collect

Use the user's chosen budgets; this is an example new experiment, not a claim
about paper results:

```bash
python skills/r5-remote-deploy/scripts/fleet.py start --machine cluster \
  --tasks L2-01 L2-01_variant01 --max-calls 150 --wall-time-s 18000
python skills/r5-remote-deploy/scripts/fleet.py logs --machine cluster
python skills/r5-remote-deploy/scripts/fleet.py status --machine cluster
python skills/r5-remote-deploy/scripts/fleet.py collect --machine cluster
```

`start` returns after launching the remote batch; the tasks run sequentially.
Results stay under `runs/remote/<hostname>/`; collection writes to
`runs/remote_collected/` locally. `stop --machine cluster` requests graceful
cleanup of that helper's owned batch. Never use broad process-name kills.

The legacy `tools/recover_l3.py` is specific to measured-state recovery bundles
and their original budgets. It is not an automatic resume mechanism for arbitrary
experiments or power-loss recordings.
