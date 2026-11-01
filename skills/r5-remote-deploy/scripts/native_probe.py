#!/usr/bin/env python3
"""One native reset/camera/planner probe on a server. Never calls GPT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--task',default='L2-02');a=p.parse_args()
out=ROOT/'runs/remote_validation'/os.uname().nodename/a.task
out.mkdir(parents=True,exist_ok=True)
(out/'result.json').unlink(missing_ok=True)
with (out/'process.log').open('wb') as log:
 result=subprocess.run([os.environ['GPT_BENCHMARK_PYTHON'],'-m','robot_tasks.integrationtest.diagnostics.scene_smoke',a.task,'--plan-probe','--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
# Kit fast shutdown can report exit 0 even when the diagnostic failed.
report=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
passed=result.returncode==0 and report.get('passed') is True
print('NATIVE_PROBE', 'PASS' if passed else 'FAIL',out,flush=True)
raise SystemExit(0 if passed else 2)
