<div align="center">

# LLM4Robot-Bench

Part of the **LLM4Robot** research series.

**How Far Can GPT-6 Astra Go in Robotic Manipulation?**

A Hierarchical, Paired Sim–Real Benchmark

[Project Page](https://llm4robot.github.io/bench/) · [Paper](https://arxiv.org/abs/2610.34527) · [1-minute Film](https://llm4robot.github.io/bench/#overview) · [Quick Start](#quick-start) · [Benchmark](#benchmark) · [Documentation](#documentation) · [Citation](#citation)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-5.1-76B900?logo=nvidia&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

<a href="https://llm4robot.github.io/bench/media/paper/figure1.png">
  <img src="https://llm4robot.github.io/bench/media/paper/figure1.png" width="1040" alt="Figure 1: paired sim–real evaluation, the three-level LLM4Robot-Bench, and GPT-6 Astra’s success by level">
</a>

*Figure 1. Paired sim–real evaluation, increasing interaction demands, and observed performance across three levels.*

[View full-size figure](https://llm4robot.github.io/bench/media/paper/figure1.png) · [Figure PDF](https://llm4robot.github.io/bench/media/paper/figure1.pdf) · [Explore the interactive project page](https://llm4robot.github.io/bench/)

</div>

## Overview

LLM4Robot-Bench evaluates contact-rich manipulation across **16 tasks**, **three levels** of interaction difficulty, and paired simulation and real-world setups on the **ARX R5** dual-arm platform.

This repository provides the simulation framework, task assets, configurations, and automatic evaluators. Each task has an independently selectable original and fixed alternative configuration, giving **32 runnable settings**. The project page brings together paired recordings, sim–real matching details, and an interactive task explorer.

<details>
<summary><strong>Watch paired experiment recordings</strong></summary>

<a href="https://llm4robot.github.io/bench/#experiments">
  <img src="https://llm4robot.github.io/bench/media/loops/overview.gif" width="900" alt="Recordings of all 16 tasks, with simulation on the left and the real robot on the right of each pair">
</a>

Explore individual clips, outcomes, and sim–real matching details on the [project page](https://llm4robot.github.io/bench/#experiments).

</details>

## Benchmark

| Level | Interaction requirement | Tasks |
| --- | --- | --- |
| **L1** | Contact selection and geometric alignment | Marker Placement · Charger Alignment · Hat Hanging · Cup Nesting · Flower Transfer |
| **L2** | Force regulation during sustained contact | Marker Cap Removal · Book Pulling · Marble Pouring · Orange Handling · Drawer Pull & Return · Jelly Handling · Whiteboard Wiping |
| **L3** | Coordinated contact reconfiguration and force regulation | Bottle Cap Removal · Nut Removal · Stacked-Book Pushing & Rotation · Block Sweeping |

Current task IDs and historical mappings are documented in the [task numbering guide](docs/release/TASK_NUMBERING.md). Original and `variant01` settings are separate experiments; alternative settings are not uniformly position-only changes. All 32 settings default to **150 model calls or 5 hours**, whichever limit is reached first. See the [evaluation protocol](docs/release/PROTOCOL.md) for the exact settings.

## Quick start

### 1. Install the framework

Use Python 3.11 and an editable source checkout:

```bash
git clone https://github.com/LLM4Robot/LLM4Robot-Bench.git
cd LLM4Robot-Bench
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

### 2. Download assets and inspect an experiment

The initial clone contains source code and documentation. Download the pinned simulation assets when needed; website media is optional for simulation. These checks and execution plans run on a CPU without calling a model:

```bash
git submodule update --init assets
python tools/verify_runtime_assets.py --usd
python tools/release/check_task_contract.py
python tools/launch.py list
python tools/reproduce.py --tasks L2-02 L2-02_variant01
```

To preview or edit the project page, download its media separately:

```bash
git submodule update --init docs/media
python -m http.server 8000 --directory docs
```

Use `git clone --recurse-submodules` to download both sets at once. GitHub’s **Download ZIP** does not include submodule contents; use a Git clone or a packaged source release with its matching asset archive, as described in [Installation](docs/release/INSTALL.md).

### L3-01: 270° preloosen configuration

The additional bottle-cap experiment uses **270° initial preloosen**, up to **150 model calls**, and a **5-hour wall-time budget**, including initialization and model waiting. Its configuration and launcher are included in this release:

```bash
python tools/experiments/bottle_preloosen.py check --experiment-config configs/experiments/l3_01_preloosen_270.json
```

This command checks the configuration on CPU without starting a simulation or making model calls. The default benchmark configuration remains available; this additional experiment does not replace the reported paper trials. See [the experiment guide](docs/release/L3_01_EXPERIMENT.md) for execution.

### 3. Run in simulation

Simulation requires **NVIDIA Isaac Sim 5.1**, a compatible **NVIDIA GPU**, the separate pinned **cuRobo** environment, and model access. Follow [Installation](docs/release/INSTALL.md) to configure these runtimes and the ignored `configs/runtime/local.toml` file.

```bash
python tools/reproduce.py --tasks L2-02_variant01 --execute
python tools/batch_run.py --summarize runs/PATH_TO_BATCH
```

Without `--execute`, `reproduce.py` prints a plan. It passes each configuration's existing seed, time limit, and call limit explicitly. Use `--suite original`, `--suite variant01`, or `--suite all` to select a complete suite. Offline dry runs check the execution lifecycle; they do not run physics or measure task success.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/release/INSTALL.md) | Python, Isaac Sim, cuRobo, model access, and local paths |
| [Evaluation protocol](docs/release/PROTOCOL.md) | Task configurations, prompts, budgets, scoring, and recorded outputs |
| [Task numbering](docs/release/TASK_NUMBERING.md) | Current IDs and mappings for historical records |
| [Validation](docs/release/VALIDATION.md) | Asset and software checks, with their scope and limitations |
| [Release guide](docs/release/RELEASE.md) | Source and asset packaging, project page, and provenance |
| [Optional remote deployment](skills/r5-remote-deploy/SKILL.md) | SSH deployment, fleet examples, and verification |

Runs retain configuration snapshots, model requests and replies, observations, actions, scores, and videos. Summaries distinguish task failures, timeouts, interruptions, and infrastructure errors. Real-robot operating conditions and records are documented separately from the simulation runtime.

## Development

The [opt-in R5 control revalidation profile](docs/release/CONTROL_REVALIDATION.md)
packages precision planning, conservative finger-mesh checks and advisory
arrival feedback without replacing frozen benchmark configurations.

```bash
python -m unittest discover -s tests -v
python tools/release/check_task_contract.py
python tools/verify_runtime_assets.py --check-git --usd
```

| Directory | Purpose |
| --- | --- |
| `src/robot_tasks/` | Shared execution, policies, planning, tasks, and evaluators |
| `configs/tasks/` | Original and alternative task configurations |
| `assets/` | Pinned [simulation assets](https://github.com/PhysLadder-Benchmark/PhysLadder-Assets); `runtime.lock.json` records SHA-256 hashes |
| `tools/release/` | Verification, packaging, and initial-state rendering |
| `docs/` | Project-page source and technical documentation |
| `docs/media/` | Optional pinned [website media](https://github.com/PhysLadder-Benchmark/PhysLadder-Media) |

For bug reports, include the task ID, commit, runtime versions, and relevant logs in a [GitHub issue](https://github.com/LLM4Robot/LLM4Robot-Bench/issues). Keep credentials, machine settings, and run outputs out of commits. Changes to prompts, scoring, physics, or initial conditions should be identified explicitly so results remain comparable.

## Citation

If you use LLM4Robot-Bench in your research, please cite our [paper](https://arxiv.org/abs/2610.34527). The BibTeX entry is also available in [CITATION.bib](CITATION.bib).

```bibtex
@misc{llm4robotbench2026,
  title = {LLM4Robot-Bench: How Far Can GPT-6 Astra Go in Robotic Manipulation? A Hierarchical, Paired Sim–Real Benchmark},
  author = {Yu, Haibao and Meng, Jiaqi and Zhang, Zijian and Jing, Bowen and Yang, Lei and Hu, Yue and Ge, Chenchen and Hou, Yiming and Xin, Chenbo and Ye, Dawen and Qin, Haifang and Li, Jin and Zhou, Weitao and Qiu, Jianing and Sun, Jiankai and Zhang, Dandan and Nie, Zaiqing and Luo, Ping and Zhao, Ding},
  year = {2026},
  eprint = {2610.34527},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  doi = {10.48550/arXiv.2610.34527},
  url = {https://arxiv.org/abs/2610.34527}
}
```

## License

Team-owned code is released under [Apache-2.0](LICENSE). Third-party material retains its original terms; see [NOTICE](NOTICE) and the [asset permission manifest](docs/release/asset_permissions.json).
