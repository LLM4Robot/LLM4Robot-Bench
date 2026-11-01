"""Diagnostic initial-pose grasp/lift/release, without GPT or task success claims.

Robot joints are placed at a solved test start only during diagnostic reset.
Closing and lifting then use physical joint drives; props are never attached.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .scene_smoke import check


def probe(backend, task, output, *, check_planner=False, hold_s=1., grasp_style="wall", place_nested=False, plan_lift=False, lift_m=.035):
    from PIL import Image
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from isaacsim.core.utils.types import ArticulationAction
    from ...robots.kinematics import R5Kinematics
    from ...robots.transforms import pose_transform
    from ...robots.control import width_to_joints

    config = task.scene.config
    kin = R5Kinematics(config.robot)
    instance = backend.articulations['left']
    names = config.robot.joint_names + config.robot.gripper.joint_names
    indices = np.array([instance.dof_names.index(n) for n in names])
    task_id = config.task_id
    item, offset, opened, closed = {
        'L1-01': ('marker', (0,0,.125), .026, .014),
        'L1-04': ('upper_cup', (-.038,0,.090), .012, .001),
        'L1-03': ('hat', (-.089,0,.072), .012, .001),
        'L1-05': ('rose', (0,0,0), .070, 0.),
    }[task_id]
    if task_id == 'L1-04' and grasp_style == 'body':
        offset, opened, closed = (0,.012,.042), .070, 0.
    initial = task.scene.measure()[item]
    target = np.array(initial['pose'][:3]) + offset
    orientation = np.array([np.sqrt(.5),0,0,-np.sqrt(.5)])
    rotation = pose_transform([*target,*orientation])[:3,:3]
    if task_id in {'L1-03','L1-04'}:
        # Vertical fingers enter from above; a horizontal finger body would
        # already intersect the curved wall before this diagnostic even closes.
        rotation = np.array([[0,1,0],[0,0,-1],[-1,0,0]])
    if task_id == 'L1-04' and grasp_style == 'body':
        rotation = pose_transform([*target,.6830127,.1830127,.1830127,-.6830127])[:3,:3]
    if task_id == 'L1-05':
        rose = pose_transform(initial['pose'])
        target = rose[:3,:3] @ np.array([0,0,.272]) + rose[:3,3] + [0,0,.005]
        # Pinch along the bud length. A crosswise pinch of its rounded surface
        # permits the long stem to pivot about the contact axis and drag on the
        # table even while the head rises. This is diagnostic geometry only.
        jaw = rose[:3,2].copy();jaw[2]=0.;jaw /= np.linalg.norm(jaw)
        approach = np.array([0.,0.,-1.])
        rotation = np.column_stack((approach,jaw,np.cross(approach,jaw)))
    def solve(position, start):
        def error(q):
            t=pose_transform(kin.forward(q,'left'))
            return np.r_[10*(t[:3,3]-position), Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec()]
        rng=np.random.default_rng(0)
        best=None
        for seed in [start]+[rng.uniform(-1.5,1.5,6) for _ in range(12)]:
            fit=least_squares(error,seed,bounds=(kin.lower,kin.upper),max_nfev=250)
            if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
            if np.linalg.norm(fit.fun)<1e-5:break
        if np.linalg.norm(best.fun)>.005:raise AssertionError('Diagnostic pose unreachable: '+str(best.fun))
        return best.x
    q=solve(target,np.zeros(6))
    planning=None
    if check_planner:
        from ...planning.client import PlannerClient
        from ...planning.scene import snapshot
        from ...types import Action, ArmCommand, ArmState
        from ...robots.control import joints_to_width
        from ...robots.transforms import transform_pose
        states={}
        for mount in config.robot.mounts:
            arm=backend.articulations[mount.name]
            idx=[arm.dof_names.index(n) for n in names]
            positions=np.asarray(arm.get_joint_positions(),dtype=float)[idx]
            states[mount.name]=ArmState(joint_positions_rad=tuple(positions[:6]),
                finger_positions_m=tuple(positions[6:]),gripper_width_m=joints_to_width(config.robot,positions[6:]))
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=target
        client=PlannerClient(config.robot,backend.settings.asset_directory/'curobo',log_directory=output)
        try:
            plan=client.plan(Action((ArmCommand('left','cartesian_pose',transform_pose(transform),'world'),),config.control_dt_s),states,snapshot(backend))
            planning=plan['diagnostics']
        finally:
            client.close()
    # Reuse production drive code, including its per-physics-step limiter.
    from ...robots.adapters import R5Adapter
    from ...config import ExecutionSettings
    controller=R5Adapter()
    controller.config, controller.backend = config, backend
    controller.instances, controller.kinematics = backend.articulations, kin
    controller.execution = ExecutionSettings.from_config(config)
    controller.reset(config.seed)
    old_hook = backend.before_physics_step
    drive_samples=[]
    def drive_tick(dt):
        controller.before_physics_step(dt)
        state=controller.state()['left']
        drive_samples.append([backend.time_s(),*state.finger_positions_m,*state.finger_velocities_m_s])
    backend.before_physics_step = drive_tick
    def command(q,width,reset=False):
        values=np.r_[q,width_to_joints(config.robot,width)]
        if reset:
            instance.set_joint_positions(values,joint_indices=indices)
            instance.set_joint_velocities(np.zeros(8),joint_indices=indices)
            controller.drive_targets['left']=tuple(values)
            if 'left' in controller.stable_grips:
                drive=controller.stable_grips['left']
                drive.reference,drive.velocity=width/2,0.
            instance.apply_action(ArticulationAction(joint_positions=values,joint_indices=indices))
        else:
            controller._start_grip('left',tuple(values[6:]),controller.state()['left'])
        controller._send_targets({**controller.targets,'left':tuple(values)})
    def step(seconds):
        for _ in range(round(seconds/config.control_dt_s)):backend.step(config.control_dt_s)
    def capture(label):
        from ...types import Observation, ArmState, Feedback
        from ...robots.transforms import transform_pose
        from ...robots.control import joints_to_width
        frames=backend.capture_rgb()
        for frame in frames:
            Image.frombytes('RGB',(frame.width,frame.height),frame.data).save(output/(label+'_'+frame.camera+'.png'))
        values=np.asarray(instance.get_joint_positions(),dtype=float)[indices]
        tcp=transform_pose(pose_transform(backend.read_ee_pose('left'))@pose_transform(config.robot.tcp_pose_wxyz))
        return Observation(backend.time_s(),frames,{'left':ArmState(
            gripper_width_m=joints_to_width(config.robot,values[6:]),
            tcp_pose_world_wxyz=tcp)},Feedback('completed'))
    command(q,opened,reset=True)
    step(.5)
    from ...grasp_guard import GraspGuard
    from ...types import Action, ArmCommand
    guard=GraspGuard(config)
    guard.accepted(Action((ArmCommand('left','gripper',(closed,)),),config.control_dt_s),capture('open'))
    before=task.scene.measure()[item]
    command(q,closed)
    step(1.)
    contact=task.scene.measure()[item]
    capture('closed')
    measured=np.asarray(instance.get_joint_positions())[indices][:6]
    current=np.array(kin.forward(measured,'left')[:3])
    lift_plan = None
    if plan_lift:
        from ...planning.client import PlannerClient
        actual = controller.state()['left']
        lift_target = tuple(np.array(actual.tcp_pose_world_wxyz[:3])+[0,0,lift_m])+tuple(actual.tcp_pose_world_wxyz[3:])
        client = PlannerClient(config.robot,backend.settings.asset_directory/'curobo',log_directory=output)
        controller.planner = client
        try:
            controller.apply(Action((ArmCommand('left','cartesian_pose',lift_target,'world'),),config.control_dt_s))
            for _ in range(240):
                feedback = controller.feedback()
                if feedback.status in {'rejected','error'}:
                    raise RuntimeError(feedback.message)
                backend.step(config.control_dt_s)
                if controller.feedback().status == 'completed':break
            actual = controller.state()['left']
            error = float(np.linalg.norm(np.array(actual.tcp_pose_world_wxyz[:3])-lift_target[:3]))
            lift_plan = {'target':lift_target,'position_error_m':error,'feedback':controller.feedback().status}
            if error > .005 or lift_plan['feedback'] != 'completed':
                raise RuntimeError('Planned test lift did not complete: '+str(lift_plan))
            lifted = np.array(actual.joint_positions_rad)
        except Exception as exc:
            (output/'lift_failure.json').write_text(json.dumps({'error':str(exc),'target':lift_target,
                'contact_forces_n':contact['fingers']['left']},indent=2))
            raise
        finally:
            client.close();controller.planner=None
    else:
        lifted=solve(current+np.array([0,0,lift_m]),measured)
        for u in np.linspace(0,1,25):
            command(measured*(1-u)+lifted*u,closed)
            backend.step(config.control_dt_s)
    hold_samples=[]
    for _ in range(round(hold_s/config.control_dt_s)):
        backend.step(config.control_dt_s)
        state=controller.state()['left']
        prop_state=task.scene.measure()[item]
        hold_samples.append(dict(time_s=backend.time_s(), fingers=state.finger_positions_m,
            finger_velocity=state.finger_velocities_m_s, tcp=state.tcp_pose_world_wxyz,
            prop_pose=prop_state['pose'],forces=prop_state['fingers']['left']))
    after=task.scene.measure()[item]
    held_observation=capture('lifted')
    vision=guard.enrich(held_observation).feedback.metrics['grasp_check']['left']
    # Contact reports are in prop-local coordinates. Compare them against the
    # original visible finger STL, using the measured link6 pose and finger q.
    import trimesh
    import xml.etree.ElementTree as ET
    from ...robots.transforms import origin_transform
    tree=ET.parse(config.robot.asset_path).getroot()
    ee=pose_transform(backend.read_ee_pose('left'))
    values=np.asarray(instance.get_joint_positions())[indices]
    prop=pose_transform(after['pose'])
    visual_errors=[]
    for i,link in enumerate(['link7','link8']):
        contacts=np.asarray(after['finger_contacts_local'][i])
        if not len(contacts):continue
        joint=next(j for j in tree.findall('joint') if j.find('child').get('link')==link)
        origin=joint.find('origin')
        t=origin_transform(tuple(map(float,origin.get('xyz').split())),
                           tuple(map(float,origin.get('rpy').split())))
        t[:3,3]+=t[:3,:3]@(np.array(list(map(float,joint.find('axis').get('xyz').split())))*values[6+i])
        inverse=np.linalg.inv(ee@t)
        world=contacts@prop[:3,:3].T+prop[:3,3]
        local=world@inverse[:3,:3].T+inverse[:3,3]
        mesh=trimesh.load(config.robot.asset_path.parent.parent/'meshes'/(link+'.STL'),force='mesh')
        _,distance,_=trimesh.proximity.closest_point(mesh,local)
        visual_errors.extend(distance.tolist())
    placement=[]
    if place_nested:
        if task_id != 'L1-04':
            raise ValueError('Nested placement diagnostic is only for L1-04')
        from ...planning.client import PlannerClient
        client=PlannerClient(config.robot,backend.settings.asset_directory/'curobo',log_directory=output)
        controller.planner=client
        try:
            measured=controller.state()['left'].tcp_pose_world_wxyz
            delta=np.array(measured[:3])-np.array(task.scene.measure()[item]['pose'][:3])
            lower=np.array(task.scene.measure()['lower_cup']['pose'][:3])
            high=lower[2]+.1076+.020+delta[2]
            goals=[[*measured[:2],high], [lower[0]+delta[0],lower[1]+delta[1],high],
                   [lower[0]+delta[0],lower[1]+delta[1],lower[2]+.1076-.012+delta[2]]]
            for xyz in goals:
                controller.apply(Action((ArmCommand('left','cartesian_pose',tuple(xyz)+tuple(measured[3:]),'world'),),config.control_dt_s))
                for _ in range(120):
                    feedback=controller.feedback()
                    if feedback.status in {'rejected','error'}:
                        raise RuntimeError(feedback.message)
                    backend.step(config.control_dt_s)
                    if controller.feedback().status=='completed':break
                state=controller.state()['left']
                error=float(np.linalg.norm(np.array(state.tcp_pose_world_wxyz[:3])-xyz))
                placement.append({'target':xyz,'position_error_m':error,'feedback':controller.feedback().status})
                if error>.005:raise RuntimeError('Placement diagnostic did not reach target')
            lifted=np.array(controller.state()['left'].joint_positions_rad)
            held_observation=capture('entered')
        except Exception as exc:
            (output/'placement_failure.json').write_text(json.dumps({'error':str(exc),'moves':placement},indent=2))
            raise
        finally:
            client.close();controller.planner=None
    guard.accepted(Action((ArmCommand('left','gripper',(opened,)),),config.control_dt_s),held_observation)
    command(lifted,opened)
    step(1.5)
    released=task.scene.measure()[item]
    release_before=guard.enrich(capture('released')).feedback.metrics.get('release_check',{}).get('left')
    # Diagnostic 5-mm probe only, never a prescribed GPT task trajectory.
    actual=controller.state()['left']
    q_probe=solve(np.array(actual.tcp_pose_world_wxyz[:3])+[0,0,.005],np.array(actual.joint_positions_rad))
    for u in np.linspace(0,1,12):
        command(np.array(actual.joint_positions_rad)*(1-u)+q_probe*u,opened)
        backend.step(config.control_dt_s)
    step(.5)
    release_after=guard.enrich(capture('release_probe')).feedback.metrics.get('release_check',{}).get('left')
    rise=after['pose'][2]-before['pose'][2]
    stable_samples=hold_samples[-max(1,round(min(hold_s,5)/config.control_dt_s)):]
    positions=np.array([v['fingers'] for v in stable_samples])
    speeds=np.abs(np.array([v['finger_velocity'] for v in stable_samples]))
    drift=np.ptp(np.array([v['prop_pose'][:3] for v in stable_samples]),axis=0)
    stability={'hold_s':hold_s,'finger_span_m':np.ptp(positions,axis=0).tolist(),
               'max_center_offset_m':float(np.max(np.abs(positions[:,0]-positions[:,1]))/2),
               'max_finger_speed_m_s':float(speeds.max()),'prop_span_m':drift.tolist()}
    (output/'hold_samples.json').write_text(json.dumps(hold_samples))
    report={'task_id':task_id,'gpt_called':False,'scope':'diagnostic reset then drive-only grip/lift/release',
            'rise_m':rise,'bilateral_force_after_lift_n':after['fingers']['left'],
            'release_drop_m':after['pose'][2]-released['pose'][2],
            'max_contact_distance_to_visible_finger_m':max(visual_errors,default=None),
            'planning_from_home':planning, 'grasp_style':grasp_style,'stability':stability,
            'placement':placement, 'release_before_probe':release_before,'release_after_probe':release_after,
            'actual_finger_limits':backend.scene_report['arms']['left'].get('actual_finger_limits'),
            'visual_grasp_evidence':vision,
            'before':before,'contact':contact,'after':after,'released':released,
            'passed':bool(vision['status']=='verified' and visual_errors and max(visual_errors)<.001 and
                          max(before['fingers']['left'])<.05 and rise>.020 and min(after['fingers']['left'])>.02 and
                          max(released['fingers']['left'])<.05)}
    report['max_measured_finger_speed_all_m_s']=float(np.max(np.abs(np.array(drive_samples)[:,3:5])))
    report['control_passed']=bool(rise>.020 and min(after['fingers']['left'])>.02
        and max(stability['finger_span_m'])<.001 and stability['max_center_offset_m']<.001
        and stability['max_finger_speed_m_s']<.033)
    report['lift_plan'] = lift_plan
    report['requested_lift_m'] = lift_m
    report['release_physically_complete']=bool(max(released['fingers']['left'])<.05)
    (output/'grasp.json').write_text(json.dumps(report,indent=2))
    print('NATIVE_GRASP',task_id,report['passed'],'rise',rise,'forces',after['fingers']['left'],flush=True)
    # Restore both test participants before scene_smoke checks initial scoring.
    command(np.zeros(6),sum(config.robot.gripper.width_limits_m)/2,reset=True)
    task.reset(backend,config.seed)
    step(2.)
    backend.before_physics_step = old_hook
    return {k:report[k] for k in ['passed','rise_m','bilateral_force_after_lift_n','release_drop_m']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['L1-01','L1-03','L1-04','L1-05'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-planner',action='store_true')
    parser.add_argument('--place-nested',action='store_true')
    parser.add_argument('--plan-lift',action='store_true',help='Execute the held-object lift through the production planner and controller')
    parser.add_argument('--lift-m',type=float,default=.035,help='Diagnostic lift height, never supplied as a GPT action')
    parser.add_argument('--hold-s',type=float,default=1.)
    parser.add_argument('--grasp-style',choices=['wall','body'],default='wall')
    args=parser.parse_args()
    if not np.isfinite(args.hold_s) or args.hold_s < .1:
        parser.error("--hold-s must be finite and at least 0.1 seconds")
    if not np.isfinite(args.lift_m) or not .02 <= args.lift_m <= .35:
        parser.error("--lift-m must be finite and between 0.02 and 0.35 metres")
    code=check(args.task,args.output,prepare=False,depth=True,
               capture_hook=lambda backend,task,output:probe(backend,task,output,check_planner=args.check_planner,hold_s=args.hold_s,grasp_style=args.grasp_style,place_nested=args.place_nested,plan_lift=args.plan_lift,lift_m=args.lift_m))
    report=args.output/'grasp.json'
    return 0 if code==0 and report.exists() and json.loads(report.read_text())['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
