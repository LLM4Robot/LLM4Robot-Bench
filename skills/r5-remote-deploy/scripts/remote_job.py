#!/usr/bin/env python3
"""One owned benchmark job per worker, with process-identity checked stop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STATE=ROOT/'runs'/'remote'/os.uname().nodename


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def current():
    p=STATE/'job.json'
    d=json.loads(p.read_text()) if p.exists() else {}
    d['alive']=bool(d.get('pid') and identity(d['pid'])==d.get('identity'))
    return d


def save(data):
    p=STATE/'job.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def require_other_deployment_idle():
    """An isolated directory is not permission to double-book the same GPU."""
    for name in json.loads(os.environ.get('GPT_BENCHMARK_GUARD_PROJECTS', '[]')):
        project = Path(name).resolve()
        if project == ROOT:
            continue
        path = project / 'runs/remote' / os.uname().nodename / 'job.json'
        if not path.exists():
            continue
        job = json.loads(path.read_text())
        if job.get('pid') and identity(job['pid']) == job.get('identity'):
            raise SystemExit(f'This worker is busy in {project.name}: {job.get("tasks")}; choose an idle worker')


def check_runtime():
    """Check actual child interpreters before accepting a job; no GPU or GPT calls."""
    checks = (
        ('Isaac', 'GPT_BENCHMARK_PYTHON',
         'import importlib.util; import numpy, PIL; '
         'assert importlib.util.find_spec("isaacsim"), "isaacsim package is missing"'),
        ('planner', 'GPT_BENCHMARK_PLANNER_PYTHON', 'import torch, curobo'),
    )
    for name, variable, code in checks:
        executable = os.environ.get(variable)
        if not executable or not Path(executable).is_file():
            raise SystemExit(f'{name} runtime missing: {variable}={executable!r}; run fleet.py bootstrap')
        try:
            result = subprocess.run([executable, '-c', code], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(f'{name} runtime preflight failed: {exc}') from exc
        if result.returncode:
            raise SystemExit(f'{name} runtime preflight failed (exit {result.returncode}):\n'
                             + (result.stdout + result.stderr)[-3000:])
    print('REMOTE_RUNTIME_READY', flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status','stop','logs']);p.add_argument('--tasks',nargs='+');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-calls',type=int);p.add_argument('--wall-time-s',type=int)
    p.add_argument('--experiment-config', type=Path)
    a=p.parse_args();STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'control.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        d=current()
        if a.action=='status':print(json.dumps(d,indent=2));return
        if a.action=='logs':
            if not d:raise SystemExit('No recorded job')
            os.execvp('tail',['tail','-n','40','-F',d['log']])
        if a.action=='stop':
            if d['alive']:
                # Batch handles SIGINT and reaps only its own Isaac/planner children.
                os.kill(d['pid'],signal.SIGINT)
                deadline=time.monotonic()+50
                while identity(d['pid'])==d['identity'] and time.monotonic()<deadline:time.sleep(.5)
                if identity(d['pid'])==d['identity']:raise SystemExit('Stop requested; cleanup still running. Inspect the log before retrying.')
            print(json.dumps(current(),indent=2));return
        if d['alive']:raise SystemExit('This worker already has an owned benchmark job')
        if not a.tasks:raise SystemExit('--tasks is required')
        experiment = None
        if a.experiment_config:
            experiment = (ROOT/a.experiment_config).resolve()
            if not experiment.is_relative_to(ROOT/'configs/experiments'):
                raise SystemExit('Experiment configuration must be inside configs/experiments')
            import importlib.util
            loader=importlib.util.spec_from_file_location('bottle_experiment',ROOT/'tools/experiments/bottle_preloosen.py')
            module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
            selected=module.load_experiment(experiment)
            if a.tasks != [selected['base_task']]:
                raise SystemExit('Experiment requires exactly its declared base task')
            if not a.dry_run:module.require_idle_gpu(selected['gpu_index'])
        require_other_deployment_idle()
        if not a.dry_run:check_runtime()
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());output=STATE/f'batch_{stamp}';log=STATE/f'batch_{stamp}.log'
        command=[sys.executable,str(ROOT/'tools/batch_run.py'),'--tasks',*a.tasks,'--output',str(output)]
        if experiment:
            command=[sys.executable,str(ROOT/'tools/experiments/bottle_preloosen.py'),
                     'check' if a.dry_run else 'run','--experiment-config',str(experiment),'--output',str(output)]
        elif a.dry_run:command.append('--dry-run')
        if a.max_calls is not None:command.extend(['--max-calls',str(a.max_calls)])
        if a.wall_time_s is not None:command.extend(['--wall-time-s',str(a.wall_time_s)])
        with log.open('wb') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        d={'pid':proc.pid,'identity':identity(proc.pid),'tasks':a.tasks,'log':str(log),'output':str(output),'dry_run':a.dry_run,'started_utc':stamp}
        if experiment:d['experiment_config']=str(experiment)
        save(d);print(json.dumps(d,indent=2))

if __name__=='__main__':main()
