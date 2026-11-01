# Installation

Use a source checkout (editable installation), not a standalone wheel: scenes,
robot profiles and task configuration live beside `src/`. Python 3.11 is the
reference framework and planner version. Isaac Sim 5.1 supplies its own runtime.

## Download the required resources

A regular `git clone` downloads the framework and documentation. The runtime assets and website media are separate public submodules pinned to the release:

```bash
git submodule update --init assets       # required for simulation and asset checks
git submodule update --init docs/media   # optional: local project-page preview
```

To fetch both during cloning, use `git clone --recurse-submodules`. After changing the main-repository version, rerun the relevant `git submodule update --init` command to use its pinned resources. Do not use `--remote` for reproduction: that follows newer resource commits rather than the release pin.

GitHub’s automatic **Download ZIP** omits submodule contents. Prefer a Git clone. The separately packaged `r5-benchmark-source.tar.gz` includes the asset lock and website media; install its matching `r5-benchmark-assets.tar.gz` as shown below.

## CPU checks first

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# Only for the source-only release archive:
python tools/release/install_assets.py /path/to/r5-benchmark-assets.tar.gz
python tools/verify_runtime_assets.py --usd
python tools/release/check_task_contract.py
python -m unittest discover -s tests -v
```

If the initialized assets submodule already contains the 325 locked assets, omit installation
of the asset archive. `assets/tuojing/release.json` describes historical packages;
it is NOT the complete Combined asset delivery. Use `assets/runtime.lock.json`
and the matching release bundle. No scene rebuild is needed for normal use.

On PowerShell activate `.venv\Scripts\Activate.ps1`; the subsequent Python
commands are the same. Do not paste PowerShell's `&` syntax into Bash.

## Simulator and planner

Install NVIDIA Isaac Sim **5.1** following its [official installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html).
Use an NVIDIA GPU compatible with that release and its renderer. The FEM orange
and jelly tasks require the configured GPU physics; do not replace them with
rigid-body approximations. Asset materials depend on Isaac's built-in MDL modules.

Create the separate cuRobo environment using the pinned repository revision:

```bash
bash tools/setup_curobo.sh /absolute/path/to/runtime
```

That script pins cuRobo to `78fd485fa82d9b9a063fb4985e371814587e666a` and records
resolved dependencies. The planner and Isaac Python environments are separate. Observed versions in
the validated deployment are recorded in [reference_runtimes.json](reference_runtimes.json);
keep Isaac's bundled packages separate from framework dependencies.
For a portable Linux wrapper deployment:

```bash
export GPT_BENCHMARK_ISAAC_ROOT=/absolute/path/to/isaac-sim
export GPT_BENCHMARK_PLANNER_ROOT=/absolute/path/to/runtime/r5-curobo
export GPT_BENCHMARK_RUNTIME_ROOT=/absolute/path/to/runtime/r5-benchmark
bash tools/setup_runtime.sh --wrappers-only
export GPT_BENCHMARK_PYTHON="$GPT_BENCHMARK_RUNTIME_ROOT/bin/isaac-python"
export GPT_BENCHMARK_PLANNER_PYTHON="$GPT_BENCHMARK_RUNTIME_ROOT/bin/planner-python"
export GPT_BENCHMARK_PLANNER_REFERENCE="$PWD/assets/r5/planner_reference/geometry.json"
```

Install system rendering libraries and FFmpeg using your administrator's normal
package management. Without `--wrappers-only`, bootstrap can install Debian
rendering/video packages. It does not install GPU drivers or download Isaac.
Existing wrappers are not modified by merely editing/synchronizing source.

Alternatively copy `configs/runtime/local.example.toml` to `local.toml` and
`planner.example.toml` to `planner.local.toml`, then configure your own interpreter
and output/cache paths. Local files are ignored by Git. No developer drive or
cluster address is required.

For Windows, the existing default interpreter example is
`D:/conda/envs/isaac51/python.exe`. Set `[isaacsim].python` in `local.toml` or
`GPT_BENCHMARK_PYTHON` to your actual interpreter. `[paths].cache_dir` and
`output_dir` accept other drives and directories, for example:

```toml
[isaacsim]
python = "E:/conda/envs/isaacsim/python.exe"
[paths]
cache_dir = "E:/research_runtime/gpt_benchmark"
output_dir = "E:/experiments/runs"
```

The optional legacy `windows.example.json` template defaults to
`D:/research_runtime/gpt_benchmark`. Copy it to ignored `windows.json` only if you
need its detailed cache mapping, then edit `root` to use any absolute directory;
there is no D-drive restriction. An explicit `cache_dir` in `local.toml` takes
precedence over this legacy mapping. With neither mapping configured, cache
storage uses the operating system's per-user cache directory.

## Verify actual initial states

In an environment that can launch the simulator wrapper, on an otherwise idle
GPU, capture the actual scene and normal initialization without model calls:

```bash
export PYTHONPATH="$PWD/src"
"$GPT_BENCHMARK_PYTHON" tools/release/render_initial.py \
  L2-02_variant01 --gpu 0 --output build/initial/L2-02_variant01
```

`--gpu` is the physical Vulkan GPU index. For multi-GPU containers also set
`CUDA_VISIBLE_DEVICES` to that same GPU's UUID; the helper uses logical PhysX
GPU 0. Do not share an experiment's cache/output directory with another process.
The output contains three PNG views, reset/scene/initial-scoring checks and
planner initialization. An optional `--plan-probe` requests a small plan without
executing it. Neither mode establishes task success.

## Model and episodes

`configs/models/gpt.toml` preserves the experiment's existing model settings.
It contains no credentials. Configure access for its selected transport in your
environment, or provide `--model-config /path/to/your/model.toml`. An account must
actually have access to the configured model; the model name alone grants none.
The Responses transport expects its configured API-key environment variable.
The Codex CLI transport uses the local CLI's authenticated session. Neither
credentials nor sessions are shipped. Changing model/effort/transport is a new
experimental condition and must be recorded.

```bash
python tools/reproduce.py --tasks L2-02 L2-02_variant01  # plan only
python tools/reproduce.py --tasks L2-02_variant01 --dry-run
python tools/reproduce.py --tasks L2-02_variant01 --execute
```

All original and alternative tasks default to 150 calls or 5 hours, whichever
limit is reached first. To state these defaults explicitly:

```bash
python tools/batch_run.py --tasks L2-02_variant01 --max-calls 150 --wall-time-s 18000
```

Other explicit values override the defaults and must be reported separately;
they do not change historical experiments. A Ctrl+C
asks the owned supervisor to finalize and clean up. Hard power loss does not
guarantee a finalized MP4 or a recoverable physics state.

## Optional remote control

Local execution is the default. Optional SSH setup, deployment and model
forwarding are documented in the repository's
[remote-deployment skill](../../skills/r5-remote-deploy/SKILL.md).
Its helpers and example configuration live with that skill. The old
`tools/remote/fleet.py` command remains a compatibility entrypoint.
