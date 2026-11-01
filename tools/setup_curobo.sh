#!/usr/bin/env bash
set -euo pipefail
# Usage: bash tools/setup_curobo.sh /absolute/research/runtime/directory
runtime_root="${1:?Supply a runtime directory on the research drive}"
mkdir -p "$runtime_root/cache" "$runtime_root/tmp"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export CONDA_PKGS_DIRS="$runtime_root/cache/conda"
export TMPDIR="$runtime_root/tmp"
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=''
export PIP_INDEX_URL=https://pypi.org/simple
conda create --prefix "$runtime_root/r5-curobo" python=3.11 pip -y
if [ ! -d "$runtime_root/curobo-src/.git" ]; then
    git clone https://github.com/NVlabs/curobo.git "$runtime_root/curobo-src"
fi
git -C "$runtime_root/curobo-src" checkout 78fd485fa82d9b9a063fb4985e371814587e666a
"$runtime_root/r5-curobo/bin/python" -m pip install 'torch==2.9.1' 'cuda-core[cu12]>=0.7,<0.8' 'nvidia-cuda-nvcc-cu12==12.9.86' -e "$runtime_root/curobo-src" -e .
"$runtime_root/r5-curobo/bin/python" -m pip freeze > "$runtime_root/requirements.resolved.txt"
echo "Set configs/runtime/planner.local.toml python to $runtime_root/r5-curobo/bin/python"
