#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${GPT_BENCHMARK_RUNTIME_ROOT:-$(dirname "$project_root")/runtime_combined}"
shared_framework="${GPT_BENCHMARK_SHARED_FRAMEWORK:-}"
mkdir -p "$runtime_root"
if [[ -n "$shared_framework" && ! -e "$runtime_root/framework" && -x "$shared_framework/bin/python" ]]; then
    ln -s "$shared_framework" "$runtime_root/framework"
fi
# Install user-space rendering/video dependencies, never a kernel/GPU driver.
packages=(libegl1 libgles2 libglu1-mesa libxrandr2 libvulkan1 rsync ffmpeg)
missing=()
for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
        missing+=("$package")
    fi
done
if (( ${#missing[@]} )) && [[ "${1:-}" != --wrappers-only ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
fi
if [[ "${1:-}" == --host-only ]]; then exit 0; fi
isaac_root="${GPT_BENCHMARK_ISAAC_ROOT:?Set GPT_BENCHMARK_ISAAC_ROOT to your Isaac Sim 5.1 installation}"
planner_root="${GPT_BENCHMARK_PLANNER_ROOT:?Set GPT_BENCHMARK_PLANNER_ROOT to your cuRobo environment}"
mkdir -p "$runtime_root/bin"
if [[ ! -x "$isaac_root/python.sh" ]]; then
    echo "Isaac launcher missing: $isaac_root/python.sh; set GPT_BENCHMARK_ISAAC_ROOT before bootstrap." >&2
    exit 1
fi
if [[ ! -f "$isaac_root/VERSION" ]] || [[ "$(cat "$isaac_root/VERSION")" != 5.1.0* ]]; then
    echo "Expected an Isaac Sim 5.1.0 installation at $isaac_root" >&2
    exit 1
fi
if [[ ! -x "$planner_root/bin/python" ]]; then
    echo "Planner Python missing: $planner_root/bin/python" >&2
    exit 1
fi
if [[ ! -x "$runtime_root/framework/bin/python" ]]; then
    "${GPT_BENCHMARK_FRAMEWORK_PYTHON:-python3}" -m venv "$runtime_root/framework"
    "$runtime_root/framework/bin/python" -m pip install -e "$project_root"
fi
# Dedicated launch wrappers use the shared installed libraries without changing
# the other project's environment, launch scripts, or editable installation.
python3 - "$project_root" "$runtime_root" "$isaac_root" "$planner_root" <<'PY'
from pathlib import Path
import shlex,sys
project,runtime,isaac,planner=map(Path,sys.argv[1:])
q=shlex.quote
scripts={
 'isaac-python':f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH={q(str(project/'src'))}
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
# Some containers have an incompatible /usr/local/lib Vulkan loader. Prefer
# distro libraries for this process; Isaac adds its own bundled libs afterward.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${{LD_LIBRARY_PATH:-}}
if [[ ! -x {q(str(isaac/'python.sh'))} ]]; then
    echo 'Isaac Sim launcher is missing; rerun tools/setup_runtime.sh with a valid GPT_BENCHMARK_ISAAC_ROOT.' >&2
    exit 127
fi
exec {q(str(isaac/'python.sh'))} "$@"
''',
 'planner-python':f'''#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export PYTHONPATH={q(str(project/'src'))}
exec {q(str(planner/'bin/python'))} "$@"
'''}
for name,text in scripts.items():
 p=runtime/'bin'/name;p.write_text(text);p.chmod(0o755)
print('BOOTSTRAP_READY',project)
PY
