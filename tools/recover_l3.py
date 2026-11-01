"""Launch one measured-state L3 continuation on one explicitly selected GPU."""
import argparse, datetime, json, os, pathlib, subprocess, sys, uuid
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=[t+s for t in ['L3-01','L3-02','L2-07','L3-03','L3-04'] for s in ['', '_v3']])
    p.add_argument('--gpu',type=int,required=True,help='nvidia-smi GPU index (0..7 on the current host)')
    p.add_argument('--pack',type=pathlib.Path)
    p.add_argument('--output',type=pathlib.Path)
    p.add_argument('--probe',action='store_true',help='Restore, test hold, render three views; never call GPT')
    p.add_argument('--background',action='store_true')
    p.add_argument('--check',action='store_true',help='Validate source hashes, budgets and Vulkan without starting Isaac')
    p.add_argument('--isaac-worker',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()

def worker(a):
    from dataclasses import replace
    from types import SimpleNamespace
    from robot_tasks.recovery import Recovery
    from robot_tasks.registry import resolve_task
    from robot_tasks.execution import _gpt_policy
    from robot_tasks.backends.isaacsim import IsaacSimBackend
    from robot_tasks.robots.adapters import R5Adapter
    from robot_tasks.runner import run_episode
    recovery=Recovery(a.pack,probe=a.probe);recovery.validate_sources(ROOT)
    spec,config=resolve_task(a.task,recovery.episode.parents[4]/'configs'/(a.task+'.toml'))
    config=spec.factory.prepare_config(config)
    config=replace(config,raw={**config.raw,'simulation':{**config.raw['simulation'],'active_gpu':a.gpu,'physics_gpu':a.gpu,'headless':True}})
    if a.probe:
        class ProbePolicy:
            calls=0
            settings=SimpleNamespace(max_calls=150)
            def reset(self,seed):self.calls=0
            def act(self,context):raise AssertionError('Probe must never call a model')
        policy=ProbePolicy()
    else:policy=_gpt_policy(config,None)
    _,result=run_episode(config,IsaacSimBackend(),R5Adapter(),spec.factory(),policy,a.output/'episodes',record_video=not a.probe,recovery=recovery)
    return 0 if result.get('recovery',{}).get('validation_passed') and result['status']=='finished' else 2

def main():
    a=arguments()
    a.pack=(a.pack or ROOT/'runs/recovery_audit_20261020'/(a.task+'_recovery.json')).resolve()
    if a.isaac_worker:return worker(a)
    from robot_tasks.recovery import Recovery
    recovery=Recovery(a.pack,probe=a.probe);recovery.validate_sources(ROOT)
    if recovery.data['task']!=a.task:raise ValueError('Task/pack mismatch')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus={int(v[0]):(v[1].strip(),int(v[2])) for line in gpu_rows.splitlines() if (v:=line.split(','))}
    if a.gpu not in gpus:raise ValueError('Selected GPU does not exist')
    runtime=pathlib.Path(os.environ.get('GPT_BENCHMARK_RUNTIME_ROOT',str(ROOT.parent/'runtime_combined')));tag=os.uname().nodename+'/gpu'+str(a.gpu)
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',GPT_BENCHMARK_PERSISTENT_MODEL_RETRY='1',GPT_BENCHMARK_RUNTIME_ROOT=str(runtime),
        GPT_BENCHMARK_PLANNER_PYTHON=str(ROOT/'tools/remote/recovery_planner.sh'),GPT_BENCHMARK_RECOVERY_GPU_UUID=gpus[a.gpu][0],
        GPT_BENCHMARK_CODEX_BRIDGE_URL=os.environ.get('GPT_BENCHMARK_CODEX_BRIDGE_URL','http://127.0.0.1:18469'),GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE=os.environ.get('GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE',str(runtime/'bridge.token')),
        GPT_BENCHMARK_PLANNER_REFERENCE=str(ROOT/'assets/r5/planner_reference/geometry.json'),OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    for key,sub in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('GPT_BENCHMARK_PLANNER_CACHE','planner_cache'),('GPT_BENCHMARK_ASSET_CACHE','asset_cache')]:
        env[key]=str(runtime/sub/tag);pathlib.Path(env[key]).mkdir(parents=True,exist_ok=True)
    env['XDG_RUNTIME_DIR']=str(pathlib.Path(env['TMPDIR'])/'xdg');pathlib.Path(env['XDG_RUNTIME_DIR']).mkdir(exist_ok=True,mode=0o700)
    compat=pathlib.Path('/usr/local/cuda-13.0/compat/libcuda.so.1')
    if compat.exists():env['LD_PRELOAD']=str(compat)
    vk=subprocess.run(['vulkaninfo','--summary'],env=env,capture_output=True,text=True,timeout=30)
    if vk.returncode or 'NVIDIA' not in vk.stdout:raise RuntimeError('Vulkan preflight failed: '+vk.stderr[-1500:])
    report={'task':a.task,'gpu':a.gpu,'gpu_uuid':gpus[a.gpu][0],'effective_calls_used':recovery.used,'effective_calls_remaining':150-recovery.used,
            'estimated_remaining_wall_s':18000-recovery.used_wall,'pack':str(a.pack),'probe_only':a.probe}
    if a.check:print(json.dumps(report,ensure_ascii=False,indent=2));return 0
    if gpus[a.gpu][1]>1000:raise RuntimeError(f'GPU {a.gpu} already uses {gpus[a.gpu][1]} MiB; refusing to overlap')
    a.output=(a.output or ROOT/'runs'/('l3_recovered_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))/(a.task+'_'+uuid.uuid4().hex[:6])).resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'launch.json').write_text(json.dumps(report,indent=2))
    args=[a.task,'--gpu',str(a.gpu),'--pack',str(a.pack),'--output',str(a.output)]+(['--probe'] if a.probe else [])
    if a.background:
        with (a.output/'process.log').open('a') as f:
            p=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (a.output/'supervisor.pid').write_text(str(p.pid));print(json.dumps({**report,'pid':p.pid,'output':str(a.output),'log':str(a.output/'process.log')},ensure_ascii=False));return 0
    import fcntl
    locks=runtime/'recovery_locks'/os.uname().nodename;locks.mkdir(parents=True,exist_ok=True);owned=[]
    for name in ['gpu_'+str(a.gpu),a.task]:
        f=(locks/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);owned.append(f)
    env['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC']=str(__import__('time').monotonic()+max(0,18000-recovery.used_wall))
    cmd=[str(runtime/'bin/isaac-python'),str(pathlib.Path(__file__).resolve()),*args,'--isaac-worker']
    p=subprocess.Popen(cmd,cwd=ROOT,env=env)
    (a.output/'worker.pid').write_text(str(p.pid))
    try:
        code=p.wait(timeout=max(300,18000-recovery.used_wall+120))
    except subprocess.TimeoutExpired:
        p.terminate()
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        code=124
    (a.output/'process_result.json').write_text(json.dumps({'returncode':code,'output':str(a.output)}))
    return code

if __name__=='__main__':raise SystemExit(main())
