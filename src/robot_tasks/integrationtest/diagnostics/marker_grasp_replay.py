"""No-GPT diagnostic: reconstruct the recorded two-arm marker grasp.

Measured arm/body/cap states are restored once before stepping. Internal shoe
states were not logged and are reconstructed from their rest anchors. This is
a contact regression, not a bit-exact replay or autonomous task success.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--sequence', type=Path, help='Replay recorded commands and planned joint paths from reset')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--task', default='L2-01')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(a.state.read_text())
    from robot_tasks.backends.isaacsim import IsaacSimBackend
    from robot_tasks.registry import resolve_task
    from robot_tasks.robots.adapters import R5Adapter
    from robot_tasks.robots.kinematics import R5Kinematics
    from robot_tasks.config import ExecutionSettings
    from robot_tasks.robots.transforms import pose_transform, transform_pose
    from robot_tasks.types import Action, ArmCommand
    import numpy as np

    spec, config = resolve_task(a.task)
    config = spec.factory.prepare_config(config)
    backend, robot, task = IsaacSimBackend(), R5Adapter(), spec.factory()
    backend.run_directory = a.output
    try:
        backend.open(config)
        from pxr import UsdPhysics
        from isaacsim.core.utils.types import ArticulationAction
        robot.config, robot.backend = config, backend
        robot.execution = ExecutionSettings.from_config(config)
        robot.kinematics = R5Kinematics(config.robot)
        robot.instances = backend.add_r5(config.robot)
        backend.before_physics_step = robot.before_physics_step
        task.build_scene(backend, config)
        backend.reset(0); task.reset(backend, 0); robot.reset(0)
        if a.sequence:
            result = replay_sequence(json.loads(a.sequence.read_text()), backend, robot, task, config, a.output)
            return 0 if result['passed'] else 2
        for name, state in saved['robot'].items():
            q = tuple(state['joint_positions_rad']) + tuple(state['finger_positions_m'])
            arm, indices = robot.instances[name], robot.indices[name]
            arm.set_joint_positions(np.array(q), joint_indices=indices)
            arm.set_joint_velocities(np.array(state['joint_velocities_rad_s'] + state['finger_velocities_m_s']), joint_indices=indices)
            # The first gripper was commanded closed; restore its loaded hold.
            goal = q[:6] + ((0., 0.) if name == 'left' else q[6:])
            robot.targets[name] = robot.drive_targets[name] = goal
            arm.apply_action(ArticulationAction(joint_positions=np.array(goal), joint_indices=indices))
        for name, body in task.scene.bodies.items():
            row = saved['objects'][name]
            body.set_world_pose(np.array(row['pose'][:3]), np.array(row['pose'][3:]))
            body.set_linear_velocity(np.array(row['linear_velocity']))
            body.set_angular_velocity(np.array(row['angular_velocity']))
        base = pose_transform(saved['objects']['cap']['pose'])
        for body, _ in task.scene.aux_bodies:
            name = body.prim_path.rsplit('/', 1)[-1]
            if not name.startswith('Shoe'): continue
            joint = UsdPhysics.Joint(backend.world.stage.GetPrimAtPath('/World/L2/marker/RadialSpring' + name[-1]))
            pose = base.copy(); pose[:3, 3] = (base @ np.r_[joint.GetLocalPos0Attr().Get(), 1.])[:3]
            pose = transform_pose(pose)
            body.set_world_pose(np.array(pose[:3]), np.array(pose[3:]))
            body.set_linear_velocity(np.array(saved['objects']['cap']['linear_velocity']))
            body.set_angular_velocity(np.array(saved['objects']['cap']['angular_velocity']))
        robot.apply(Action((ArmCommand('right', 'gripper', (0.,)),), config.control_dt_s))
        rows = []; peak_speed = peak_contact = 0.
        for i in range(round(a.seconds / config.control_dt_s)):
            backend.step(config.control_dt_s)
            objects = task.scene.measure()
            frame = {n: {k: v[k] for k in ('pose', 'linear_velocity', 'fingers', 'pair_forces')} for n, v in objects.items()}
            peak_speed = max(peak_speed, *(float(np.linalg.norm(o['linear_velocity'])) for o in objects.values()))
            peak_contact = max(peak_contact, *(v for o in objects.values() for f in o['fingers'].values() for v in f))
            rows.append({'time_s':backend.time_s(), 'objects':frame})
            print('GRASP_REPLAY', i, peak_speed, peak_contact, flush=True)
            if peak_speed > 30: break
        held = all(min(rows[-1]['objects'][n]['fingers'][arm]) > .05
                   for n, arm in (('body','left'), ('cap','right')))
        result = {'gpt_called': False, 'recorded_time_s':saved['time_s'],
                  'scope':__doc__, 'task':a.task, 'peak_speed_m_s':peak_speed,
                  'peak_finger_contact_n':peak_contact,
                  'both_parts_held':held,
                  'passed':peak_speed < 2. and peak_contact < 100. and held,
                  'physics_overrides':backend.scene_report['household']['physics_overrides'], 'samples':rows}
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print('GRASP_REPLAY_RESULT', {k:v for k,v in result.items() if k not in ('samples','physics_overrides')}, flush=True)
        return 0 if result['passed'] else 2
    finally:
        backend.close()


def replay_sequence(events, backend, robot, task, config, output):
    """Replay accepted commands/trajectories with the actual dynamic robot."""
    import numpy as np
    from robot_tasks.types import Action, ArmCommand
    plans = {r['step']: r for r in events if r['event'] == 'motion_plan'}
    accepted = {r['step'] for r in events if r['event'] == 'transition'}
    class RecordedPlanner:
        step = None
        def plan(self, *args): return plans[self.step]
    robot.planner = RecordedPlanner()
    samples = []; peak_speed = peak_contact = 0.
    for row in events:
        if row['event'] == 'decision' and row['step'] in accepted:
            robot.planner.step = row['step']
            action = row['action']
            robot.apply(Action(tuple(ArmCommand(c['arm'], c['kind'], tuple(c['values']), c['frame'])
                                     for c in action['commands']), action['duration_s']))
            print('REPLAY_ACTION', row['step'], flush=True)
        elif row['event'] == 'physics':
            backend.step(config.control_dt_s)
            robot.feedback()
            objects = task.scene.measure()
            peak_speed = max(peak_speed, *(float(np.linalg.norm(o['linear_velocity'])) for o in objects.values()))
            peak_contact = max(peak_contact, *(v for o in objects.values() for f in o['fingers'].values() for v in f))
            samples.append({'time_s':backend.time_s(), 'objects':{n:{k:v[k] for k in ('pose','linear_velocity','fingers','pair_forces')} for n,v in objects.items()}})
            if row['tick'] % 24 == 0: print('REPLAY_TICK',row['tick'], peak_speed,peak_contact,flush=True)
            if peak_speed > 30: break
    result = {'gpt_called':False, 'scope':'Recorded accepted commands and cached joint paths replayed from physical reset; numerical contact history can differ.',
              'peak_speed_m_s':peak_speed,'peak_finger_contact_n':peak_contact,
              'passed':peak_speed < 2. and peak_contact < 100.,'samples':samples,
              'physics_overrides':backend.scene_report['household']['physics_overrides']}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print('SEQUENCE_RESULT',{k:v for k,v in result.items() if k not in ('samples','physics_overrides')},flush=True)
    return result


if __name__ == '__main__': raise SystemExit(main())
