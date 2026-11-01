#!/usr/bin/env python3
"""Optional remote deployment; the local benchmark does not import this module."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SKILL=ROOT/'skills/r5-remote-deploy'
SCRIPTS='skills/r5-remote-deploy/scripts'
CONFIG=Path(os.environ.get('GPT_BENCHMARK_FLEET_CONFIG',
    ROOT/'configs/runtime/remote.local.json'))
if 'GPT_BENCHMARK_FLEET_CONFIG' in os.environ and not CONFIG.is_file():
    raise FileNotFoundError(f'Remote config not found: {CONFIG}')
if not CONFIG.exists():
    CONFIG=ROOT/'tools/remote/fleet.local.json'  # Existing private deployment.
if not CONFIG.exists():
    CONFIG=SKILL/'assets/fleet.example.json'
CFG=json.loads(CONFIG.read_text(encoding='utf-8'))
STATE=Path.home()/'.local/state/gpt-benchmark-combined'
TOKEN=STATE/'bridge.token'


def ssh(machine,*args):
    port=CFG['machines'][str(machine)]
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
            *(['-J',CFG['jump_host']] if CFG.get('jump_host') and str(machine) not in CFG.get('direct_machines', []) else []),'-o','ControlMaster=auto','-o','ControlPersist=3600','-o','ControlPath=/tmp/gptbench-%C','-p',str(port),*args,CFG['user']+'@'+CFG['host']]


def remote(machine,args,**kwargs):
    return subprocess.run(ssh(machine)+[shlex.join(args)],check=True,**kwargs)


def remote_environment():
    values = {'GPT_BENCHMARK_RUNTIME_ROOT': CFG['runtime'],
              'GPT_BENCHMARK_USE_LOCAL_CODEX': '1' if CFG.get('use_local_codex', True) else '0',
              'GPT_BENCHMARK_GUARD_PROJECTS': json.dumps(CFG.get('guard_projects', []))}
    if CFG.get('use_local_codex', True):
        values.update(GPT_BENCHMARK_CODEX_BRIDGE_URL=f'http://127.0.0.1:{CFG["bridge_port"]}',
                      GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE=CFG['runtime']+'/bridge.token')
    for key, variable in (('isaac_root', 'GPT_BENCHMARK_ISAAC_ROOT'),
                          ('planner_root', 'GPT_BENCHMARK_PLANNER_ROOT')):
        if CFG.get(key):
            values[variable] = CFG[key]
    return ['env', *(f'{key}={value}' for key, value in values.items())]


def entry(*args):
    return [*remote_environment(), 'bash', CFG['project']+f'/{SCRIPTS}/remote_entry.sh', *args]


def require_idle(machine):
    # No installed runtime needed; also detects tasks started outside fleet.
    code = '''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); ancestors=set(); pid=os.getpid()
while pid and pid not in ancestors:
    ancestors.add(pid)
    try: pid=int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
    except (OSError,ValueError): break
busy=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
    if int(p.name) in ancestors: continue
    try:
        args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
        if not args or not any(s in pathlib.Path(args[0]).name.lower() for s in ('python','kit','isaac')): continue
        if (p/'cwd').resolve().is_relative_to(root) or any(v==str(root) or v.startswith(str(root)+'/') for v in args):
            busy.append(int(p.name))
    except (FileNotFoundError,ProcessLookupError): continue
    except PermissionError: raise SystemExit('Cannot inspect project processes; check permissions before deployment')
if busy: raise SystemExit('Refusing deployment: project processes are active: '+json.dumps(busy))
print('REMOTE_PROJECT_IDLE')
'''
    remote(machine, ['python3', '-c', code, CFG['project']])


def ensure_bridge():
    STATE.mkdir(parents=True,exist_ok=True);STATE.chmod(0o700)
    if not TOKEN.exists():
        fd=os.open(TOKEN,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(48)+'\n')
    sys.path.insert(0,str(ROOT/'src'))
    from robot_tasks.policies.model_config import load_gpt_settings
    from robot_tasks.policies.codex_bridge import CodexBridgeClient
    os.environ['GPT_BENCHMARK_CODEX_BRIDGE_URL']=f'http://127.0.0.1:{CFG["bridge_port"]}'
    os.environ['GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE']=str(TOKEN)
    settings=load_gpt_settings(ROOT/'configs/models/gpt.toml')
    def health():
        try:CodexBridgeClient(settings).require_chatgpt_login();return True
        except Exception:return False
    if not health():
        python=Path(sys.executable)
        with (STATE/'bridge.log').open('ab') as log:
            p=subprocess.Popen([str(python),str(SKILL/'scripts/codex_bridge_server.py'),'--token-file',str(TOKEN),'--port',str(CFG['bridge_port']),'--max-concurrent',str(CFG.get('max_concurrent',5))],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (STATE/'bridge.pid').write_text(str(p.pid))
        for _ in range(25):
            if health():break
            if p.poll() is not None:raise RuntimeError('Bridge failed; inspect '+str(STATE/'bridge.log'))
            time.sleep(.3)
        else:raise RuntimeError('Bridge startup timed out')
    print('LOCAL_CODEX_BRIDGE_READY',flush=True)


def connect(machine):
    if not CFG.get('use_local_codex', True):
        remote(machine, ['true'])
        print('CONNECTED', machine, '(model access on remote host)', flush=True)
        return
    # Private RPC token authenticates this deployment, not the OpenAI account.
    remote(machine,['bash','-c',f'umask 077; mkdir -p {shlex.quote(CFG["runtime"])}; t=$(mktemp {shlex.quote(CFG["runtime"]+"/bridge.XXXXXX")}); cat > \"$t\"; mv \"$t\" {shlex.quote(CFG["runtime"]+"/bridge.token")}'],input=TOKEN.read_bytes())
    forward=f'127.0.0.1:{CFG["bridge_port"]}:127.0.0.1:{CFG["bridge_port"]}'
    result=subprocess.run(ssh(machine,'-O','forward','-R',forward),capture_output=True,text=True)
    # An existing healthy tunnel is acceptable; otherwise surface forwarding failure.
    health=entry('-c',
            'from robot_tasks.policies.codex_bridge import create_codex_client;from robot_tasks.policies.model_config import load_gpt_settings;from pathlib import Path;create_codex_client(load_gpt_settings(Path("configs/models/gpt.toml"))).require_chatgpt_login();print("REMOTE_CODEX_BRIDGE_READY")')
    try:remote(machine,health)
    except subprocess.CalledProcessError:
        raise RuntimeError(f'Machine {machine} tunnel failed: '+result.stderr.strip())
    print('CONNECTED',machine,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['connect','sync','bootstrap','start','status','stop','logs','collect']);p.add_argument('--machine',choices=list(CFG['machines']));p.add_argument('--tasks',nargs='+');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-calls',type=int);p.add_argument('--wall-time-s',type=int)
    p.add_argument('--deployment',choices=['combined'],default='combined',
                   help='One project supports original and _variant01 task IDs (_v3 is still accepted)')
    p.add_argument('--experiment-config', help='Explicit project-relative bottle preloosen experiment JSON')
    a=p.parse_args()
    if CONFIG == SKILL/'assets/fleet.example.json':
        p.error('Configure configs/runtime/remote.local.json first; see skills/r5-remote-deploy/SKILL.md')
    if a.action in ('start','logs') and not a.machine:raise SystemExit('--machine is required for start/logs')
    machines=[a.machine] if a.machine else list(CFG['machines'])
    if a.action=='sync':
        # Mirror only the selected checkout; leave experiment outputs intact.
        m=a.machine or CFG.get('default_machine', machines[0])
        require_idle(m)
        cmd=['rsync','-az','--checksum','--delete-delay','--partial-dir=.rsync-partial','--info=progress2','--no-owner','--no-group','--safe-links','--exclude-from='+str(SKILL/'assets/rsync-excludes.txt'),'--stats','-e',shlex.join(ssh(m)[:-1]),str(ROOT)+'/',CFG['user']+'@'+CFG['host']+':'+shlex.quote(CFG['project']+'/')]
        subprocess.run(cmd,check=True);return
    if a.action=='bootstrap':
        first=a.machine or CFG.get('default_machine', 'cluster')
        require_idle(first)
        remote(first,[*remote_environment(),'bash',CFG['project']+'/tools/setup_runtime.sh'])
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda m: remote(m,[*remote_environment(),'bash',CFG['project']+'/tools/setup_runtime.sh','--host-only']),[m for m in machines if m!=first]))
        return
    if a.action in ('connect','start') and not a.dry_run:
        if CFG.get('use_local_codex', True):ensure_bridge()
        with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(connect,machines))
    if a.action=='connect':return
    if a.action in ('start','logs') and not a.machine:raise SystemExit('--machine is required for start/logs')
    if a.action=='collect':
        dest=ROOT/'runs'/'remote_collected';dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','-e',shlex.join(ssh(a.machine or CFG.get('default_machine','cluster'))[:-1]),CFG['user']+'@'+CFG['host']+':'+shlex.quote(CFG['project']+'/runs/remote/'),str(dest)+'/'],check=True);return
    for machine in machines:
        args=entry(CFG['project']+f'/{SCRIPTS}/remote_job.py',a.action)
        if a.tasks:args+=['--tasks',*a.tasks]
        if a.dry_run:args+=['--dry-run']
        if a.max_calls is not None:args+=['--max-calls',str(a.max_calls)]
        if a.wall_time_s is not None:args+=['--wall-time-s',str(a.wall_time_s)]
        if a.experiment_config:args+=['--experiment-config',a.experiment_config]
        print('MACHINE',machine,flush=True);remote(machine,args)

if __name__=='__main__':main()
