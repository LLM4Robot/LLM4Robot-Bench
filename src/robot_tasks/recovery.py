"""Reconstruct an interrupted L3 episode from measured evidence, never from GPT guesses."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .types import ArmState, Assessment, Feedback, Observation, PolicyInput, RGBFrame


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_attributes(target, saved):
    for key, value in saved.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown recovery field: {type(target).__name__}.{key}")
        old = getattr(target, key)
        if dataclasses.is_dataclass(old):
            restore_attributes(old, value)
            continue
        if isinstance(old, np.ndarray):
            value = np.asarray(value, dtype=old.dtype)
            if value.shape != old.shape:
                raise ValueError(f"Recovery array shape mismatch: {key}")
        elif isinstance(old, set):
            value = set(value)
        elif key == 'previous' and isinstance(value, list) and isinstance(target, object):
            # Wipe anchor is an xyz vector. NutProgress.previous is a tuple.
            value = np.asarray(value) if hasattr(target, 'coverage') else tuple(value)
        setattr(target, key, value)


class Recovery:
    def __init__(self, path, *, probe=False):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text())
        self.probe = probe
        self.episode = Path(self.data['source_episode'])
        self.snapshot = self.data['snapshot_event']
        self.time = self.snapshot.get('time_s', self.snapshot.get('observation', {}).get('time_s'))
        self.robot_saved = self.snapshot.get('robot', self.snapshot.get('observation', {}).get('robot'))
        self.objects = self.snapshot['evaluation']['metrics']['objects']
        self.used = self.data['used_effective_calls']
        self.used_wall = self.data['wall_budget']['estimated_used_s']
        self.report = {'protocol': 'measured_state_reconstruction_v1', 'source_episode': str(self.episode),
                       'source_episode_sha256': self.data['episode_sha256'], 'probe_only': probe,
                       'restored_effective_calls': self.used, 'original_call_limit': 150,
                       'estimated_prior_wall_time_s': self.used_wall, 'outage_time_excluded': True,
                       'hardware_migration': 'L20 to RTX 4090 D', 'bitwise_continuity': False,
                       'incomplete_action_policy': 'cancel old trajectory, hold measured arm pose and retained grip reference, then request a new model decision',
                       'validation_passed': False}
        validation = self.data['validation']
        if validation['total_mismatches'] or not validation['asset_hashes_match']:
            raise ValueError('Recovery evidence validation failed')
        if sha256(self.episode/'episode.jsonl') != self.data['episode_sha256']:
            raise ValueError('Source episode changed since recovery extraction')
        if self.used >= 150 or self.used_wall >= 18000:
            raise ValueError('Original experiment budget exhausted')
        self.events = []
        with (self.episode/'episode.jsonl').open() as f:
            for line in f:
                event = json.loads(line)
                if event.get('event') != 'physics':
                    self.events.append(event)
        accepted = [e for e in self.events if e.get('event') == 'model_exchange' and e['exchange']['status'] == 'accepted']
        if len(accepted) != self.used:
            raise ValueError('Effective call count differs from recovery bundle')
        self.total_calls = max((e['exchange']['call'] for e in self.events if e.get('event') == 'model_exchange'), default=0)
        self.failures = self.total_calls - self.used
        self.initial_observation = next(e['observation'] for e in self.events if e.get('event') == 'reset')

    def validate_sources(self, root):
        root = Path(root)
        manifest = json.loads((self.episode/'manifest.json').read_text())
        change_file = root/'tools/remote/recovery_patch_manifest.json'
        changes = json.loads(change_file.read_text()) if change_file.exists() else {}
        for name, expected in manifest['source_sha256'].items():
            current = sha256(root/'src/robot_tasks'/name)
            if current == expected:
                continue
            approved = changes.get(name, {})
            if approved.get('before') != expected or approved.get('after') != current:
                raise ValueError(f'Unverified source change since interrupted episode: {name}')
        for row in self.data['scene']['household']['assets']:
            if sha256(root/'assets/tuojing'/row['path']) != row['sha256']:
                raise ValueError('Recovery asset changed: '+row['path'])

    def restore_history(self, policy):
        if self.probe:
            policy.calls = self.total_calls
            policy.generation_failure_calls = self.failures
            return
        current = None
        pairs = []
        for event in self.events:
            if 'observation' in event and event.get('event') != 'grasp_reference':
                current = event['observation']
            if event.get('event') == 'model_exchange' and event['exchange']['status'] == 'accepted':
                if current is None:
                    raise ValueError('Missing observation preceding accepted response')
                pairs.append((current, event))
        from .policies.sparse_action import orientation_degrees
        from .robots.control import width_to_normalized
        from PIL import Image
        for index, (obs, event) in enumerate(pairs):
            robot = {name: ArmState(**{k: tuple(v) if isinstance(v,list) else v for k,v in row.items()}) for name,row in obs['robot'].items()}
            if index >= len(pairs)-policy.settings.history_steps:
                frames=[]
                for frame in obs['rgb']:
                    if frame.get('depth'):
                        raise ValueError('RGB-D history recovery is not implemented; refusing to drop depth')
                    p=(self.episode/frame['path']).resolve()
                    if not p.is_relative_to(self.episode.resolve()):raise ValueError('Frame outside source episode')
                    with Image.open(p) as im:
                        image=im.convert('RGB');data=image.tobytes()
                    frames.append(RGBFrame(frame['camera'],frame['time_s'],frame['width'],frame['height'],data,frame.get('calibration')))
                context=PolicyInput(policy.config.instruction,Observation(obs['time_s'],tuple(frames),robot,Feedback(**obs['feedback'])),event['step'])
                message,_,_,_=policy.observation_message(context)
            else:
                states={name:dataclasses.asdict(row) for name,row in robot.items()}
                for name,row in robot.items():
                    states[name]['gripper_normalized_measured']=None if row.gripper_width_m is None else width_to_normalized(policy.config.robot,row.gripper_width_m)
                text={'instruction':policy.config.instruction,'step':event['step'],'simulation_time_s':obs['time_s'],'robot':states,'feedback':obs['feedback'],
                      'tcp_orientation_reference_deg':{name:dict(zip(('roll_deg','pitch_deg','yaw_deg'),orientation_degrees(row.tcp_pose_world_wxyz))) for name,row in robot.items() if row.tcp_pose_world_wxyz}}
                content=[{'type':'input_text','text':json.dumps(text)}]
                for frame in event['exchange']['frames']:
                    content.append({'type':'input_text','text':json.dumps({k:v for k,v in frame.items() if k not in ('original_size','encoded_size','png_bytes','sha256','visual_aids')})})
                message={'role':'user','content':content}
            policy.remember(message,event['exchange']['response_text'])
        policy.calls=self.total_calls
        policy.generation_failure_calls=self.failures
        self.report['restored_history_pairs']=len(pairs)

    def restore(self, backend, robot, task, policy, role_monitor):
        from isaacsim.core.utils.types import ArticulationAction
        if task.scene.aux_bodies:
            raise ValueError('Unrecorded auxiliary bodies prevent restoration')
        if set(task.scene.bodies)!=set(self.objects):
            raise ValueError('Tracked object set changed')
        for name, body in task.scene.bodies.items():
            row=self.objects[name];pose=np.asarray(row['pose'])
            body.set_world_pose(position=pose[:3],orientation=pose[3:])
            if name not in task.scene.fixed_names:
                body.set_linear_velocity(np.asarray(row['linear_velocity']))
                body.set_angular_velocity(np.asarray(row['angular_velocity']))
        self.old_held=[]
        for name,row in self.objects.items():
            for arm,forces in row['fingers'].items():
                if min(forces)>.05:self.old_held.append((name,arm))
        robot.paths.clear();robot.pending_grippers.clear();robot.motion_arms.clear();robot.cartesian_targets.clear()
        robot.rejection=None;robot.collision_stop=None
        feedback=self.snapshot.get('feedback',self.snapshot.get('observation',{}).get('feedback',{}))
        self.applied_targets={}
        for name,row in self.robot_saved.items():
            arm=robot.instances[name];indices=robot.indices[name]
            q=np.asarray(row['joint_positions_rad']+row['finger_positions_m']);v=np.asarray(row['joint_velocities_rad_s']+row['finger_velocities_m_s'])
            if q.shape!=(8,) or v.shape!=(8,) or not np.isfinite([q,v]).all():raise ValueError('Incomplete/nonfinite robot state')
            arm.set_joint_positions(q,joint_indices=indices);arm.set_joint_velocities(v,joint_indices=indices)
            ref=feedback.get('metrics',{}).get('arms',{}).get(name,{}).get('drive_reference_fingers_m',row['finger_positions_m'])
            # Cancel unfinished motion, retain grip drive reference; no automatic release/regrasp.
            target=tuple(map(float,q[:6]))+tuple(map(float,ref))
            robot.targets[name]=robot.drive_targets[name]=target
            if name in robot.stable_grips:
                robot.stable_grips[name].reference=float(np.mean(ref));robot.stable_grips[name].velocity=0.
            arm.apply_action(ArticulationAction(joint_positions=np.asarray(target),joint_indices=indices))
            self.applied_targets[name]=target
        raw_clock=backend.time_s
        self.clock_base=raw_clock
        backend.time_s=lambda: self.time+raw_clock()-self.clock_origin
        self.clock_origin=raw_clock()
        restore_attributes(task.evaluator,self.data['evaluator'])
        if hasattr(task,'observer_monitor'):restore_attributes(task.observer_monitor,self.data['observer_monitor'])
        if self.data['ink_ledger']:
            restore_attributes(task.scene.ledger,self.data['ink_ledger']);task.scene._render_ink()
        if hasattr(task,'_last_evaluation_time'):
            task._last_evaluation_time=self.time;task._last_assessment=Assessment(**self.snapshot['evaluation'])
        if hasattr(task,'_last_time'):
            task._last_time=self.time;task._assessment=Assessment(**self.snapshot['evaluation'])
        role_monitor.reference={a:tuple(self.initial_observation['robot'][a]['joint_positions_rad']) for a in role_monitor.arms}
        role_monitor.max_drift=self.snapshot['evaluation']['metrics'].get('fixed_arm_max_drift_rad',0.)
        self.restore_history(policy)
        self.readback=robot.state()
        joint_error=max(float(np.max(np.abs(np.asarray(self.readback[a].joint_positions_rad)-np.asarray(row['joint_positions_rad'])))) for a,row in self.robot_saved.items())
        pose_error=max(float(np.linalg.norm(np.asarray(task.scene.bodies[n].get_world_pose()[0])-np.asarray(row['pose'][:3]))) for n,row in self.objects.items())
        self.report.update(joint_readback_max_error_rad=joint_error,object_readback_max_position_error_m=pose_error,
                           source_step=self.data['last_decision']['step'],source_boundary_step=self.data['last_boundary']['step'])
        if joint_error>1e-5 or pose_error>1e-5:raise ValueError('Restored physical state readback mismatch')
        # Observe a bounded hold. Do not credit its unintended contact motion as task progress.
        before={n:np.asarray(b.get_world_pose()[0]) for n,b in task.scene.bodies.items()}
        for _ in range(12):backend.step(robot.config.control_dt_s)
        from .tasks.bench.native_l2 import NativeL2Scene
        task.scene.cached_time=None
        measured=NativeL2Scene.measure(task.scene)  # Bypass ink wear during recovery calibration.
        shifts={n:float(np.linalg.norm(np.asarray(row['pose'][:3])-before[n])) for n,row in measured.items()}
        holds={f'{n}:{a}':bool(min(measured[n]['fingers'][a])>.05) for n,a in self.old_held}
        self.report.update(hold_validation_s=12*robot.config.control_dt_s,object_displacement_m=shifts,retained_bilateral_contacts=holds)
        if any(not x for x in holds.values()) or max(shifts.values(),default=0)>.03:
            raise ValueError('Recovered hold unstable; see recovery report; model was not called')
        self.clock_origin=raw_clock()  # Do not spend task simulation time on recovery calibration.
        task.scene.cached_time=None;task.scene.cached_rows=None
        if self.data['ink_ledger']:
            task.scene.ledger.previous=None;task.scene.ledger.previous_gain=0.
        # New contact measurements are obtained on the next physics tick.
        self.report['validation_passed']=True
        return Feedback('recovered','Simulation was restored after interruption. Any unfinished old trajectory was cancelled. Arms hold their restored pose and gripper references. Inspect the current images and measured robot state before issuing the next action.',{'source_step':self.data['last_decision']['step'],'previous_action_interrupted':self.data['last_decision']['step']>self.data['last_boundary']['step'],'remaining_decisions':150-self.used})

    def record(self, recorder):
        (recorder.directory/'recovery.json').write_text(json.dumps(self.report,indent=2),encoding='utf8')
        manifest_path=recorder.directory/'manifest.json'
        m=json.loads(manifest_path.read_text());m['episode_isolation']='continuation from recorded measurements; original evidence immutable';m['recovery']=self.report
        manifest_path.write_text(json.dumps(m,indent=2),encoding='utf8')
        recorder.event({'event':'recovery','details':self.report})
