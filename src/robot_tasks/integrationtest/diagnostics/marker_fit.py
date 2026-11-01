"""No-GPT L2-01 hollow-cap regression on the delivered marker and box.

Passive drops and an explicitly guided axial pull fixture, not autonomous task
success. The fixture holds the barrel and guides the cap's axis; forces, passive
springs and barrel/cap collision determine whether it can separate.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def run_case(name, fixed, hz, seconds, recorded=None):
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim, RigidPrim
    from isaacsim.core.utils.stage import create_new_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from robot_tasks.paths import PROJECT_ROOT
    from robot_tasks.video_delivery import scenes
    from robot_tasks.robots.transforms import pose_transform, transform_pose
    from robot_tasks.tasks.bench.marker_contacts import refresh_delivered_groups

    World.clear_instance(); create_new_stage()
    dt = 1 / hz
    world = World(stage_units_in_meters=1, physics_dt=dt, rendering_dt=1/12,
                  backend='numpy', device='cpu',
                  sim_params={'use_gpu_pipeline':False, 'use_fabric':False})
    physics = world.get_physics_context()
    physics.enable_gpu_dynamics(False); physics.set_solver_type('TGS')
    scene = PhysxSchema.PhysxSceneAPI.Apply(physics.get_current_physics_scene_prim())
    scene.CreateEnableCCDAttr(True); scene.CreateEnableStabilizationAttr(False)
    scene.CreateMinPositionIterationCountAttr(32)
    scene.CreateMinVelocityIterationCountAttr(0)
    stage = world.stage
    row = scenes()['L2-01']; source = str(PROJECT_ROOT/'assets/tuojing'/row['runtime_path'])
    marker = '/World/L2/marker'; box = '/World/L2/support_box'
    for path in (marker, box):
        stage.DefinePrim(path).GetReferences().AddReference(source, path)
    UsdGeom.Xformable(stage.GetPrimAtPath(marker)).MakeMatrixXform().Set(Gf.Matrix4d(1.))
    world.scene.add(FixedCuboid(prim_path='/World/Table',name='table',
        position=np.array([.12, 1.12, .83]),scale=np.array([2.,2.,.1]),size=1.))
    # Fixture transforms use the delivered global positions even if a later
    # source scene gains a transformed parent.
    for key in ('body','cap'):
        prim=stage.GetPrimAtPath(row['tracked'][key]['path'])
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(pose_transform(row['tracked'][key]['pose']).T.tolist()))
    box_pose=row['fixtures'][0]['pose']
    UsdGeom.Xformable(stage.GetPrimAtPath(box)).MakeMatrixXform().Set(Gf.Matrix4d(pose_transform(box_pose).T.tolist()))
    # Shoes have marker-local transforms. Reconstruct their world pose from the
    # cap's rest anchors, not from recorded simulation state.
    base=pose_transform(row['tracked']['cap']['pose'])
    poses={'Body':base.copy(), 'Cap':base.copy()}
    if name == 'recorded_release':
        if recorded is None:
            raise ValueError('recorded_release requires --recorded-state')
        poses = {key.title():pose_transform(recorded[key]['pose']) for key in ('body','cap')}
        base = poses['Cap'].copy()
    for i in range(4):
        j=UsdPhysics.Joint(stage.GetPrimAtPath(marker+'/RadialSpring'+str(i)))
        m=base.copy();m[:3,3]=(base@np.r_[j.GetLocalPos0Attr().Get(),1])[:3]
        poses['Shoe'+str(i)]=m
    delta={'rest':(0,0,0), 'centre_drop':(0,0,.03),
           'rim_drop':(0,-.09,.03), 'corner_drop':(.06,-.09,.03),
           'axial_pull':(0,0,.08), 'recorded_release':(0,0,0)}[name]
    for n,m in poses.items():
        m[:3,3]+=delta
        UsdGeom.Xformable(stage.GetPrimAtPath(marker+'/'+n)).MakeMatrixXform().Set(Gf.Matrix4d(m.T.tolist()))
    override=refresh_delivered_groups(stage,marker,simplify=fixed)
    if name=='axial_pull':
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(marker+'/Body')).CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(marker+'/Body')).CreateEnableCCDAttr(False)
        joint=UsdPhysics.PrismaticJoint.Define(stage,'/World/TestAxialGuide')
        joint.CreateBody0Rel().SetTargets([marker+'/Body']);joint.CreateBody1Rel().SetTargets([marker+'/Cap'])
        joint.CreateAxisAttr('Z');joint.CreateCollisionEnabledAttr(True)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,.127));joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,.127))
    bodies={n:world.scene.add((SingleXFormPrim if name=='axial_pull' and n=='Body' else SingleRigidPrim)(
        marker+'/'+n,name=n)) for n in poses}
    forces=RigidPrim(marker+'/Cap',name='force_view',reset_xform_properties=False)
    world.reset();forces.initialize()
    if name == 'recorded_release':
        for n in ('Body','Cap'):
            bodies[n].set_linear_velocity(np.array(recorded[n.lower()]['linear_velocity']))
            bodies[n].set_angular_velocity(np.array(recorded[n.lower()]['angular_velocity']))
        for i in range(4):
            bodies['Shoe'+str(i)].set_linear_velocity(np.array(recorded['cap']['linear_velocity']))
            bodies['Shoe'+str(i)].set_angular_velocity(np.array(recorded['cap']['angular_velocity']))
    samples=[]; peak=0.; separated=False; max_force=0.
    axis=base[:3,2]; start=bodies['Cap'].get_world_pose()[0].copy()
    try:
        for i in range(round(seconds*hz)):
            t=i*dt
            if name=='axial_pull':
                displacement=float(np.dot(bodies['Cap'].get_world_pose()[0]-start,axis))
                separated=separated or displacement>.035
                # A finite-stiffness test gripper moves at 4 cm/s. A constant
                # 4 N load on a released 4.6 g cap would legitimately accelerate
                # it to ~7 m/s over 3 cm, obscuring numerical-launch detection.
                target=min(.08,max(0.,t-1)*.04)
                velocity=float(np.dot(bodies['Cap'].get_linear_velocity(),axis))
                force=0. if separated else (.03 if t<1 else float(np.clip(
                    200*(target-displacement)-2*velocity,-4.,4.)))
                forces.apply_forces(np.array([axis*force],dtype=np.float32))
                max_force=max(abs(force),max_force)
            world.step(render=False)
            speeds={n:(0. if name=='axial_pull' and n=='Body' else float(np.linalg.norm(b.get_linear_velocity())))
                    for n,b in bodies.items()}
            peak=max(peak,max(speeds.values()))
            if (i+1)%round(hz/60):continue
            p,q=bodies['Cap'].get_world_pose()
            samples.append({'time_s':(i+1)*dt,'speeds_m_s':speeds,'cap_pose':[*map(float,p),*map(float,q)],
                'axial_displacement_m':float(np.dot(p-start,axis))})
            if not np.isfinite(p).all() or peak>50:break
        tail=samples[-60:]
        speed=max(max(s['speeds_m_s'].values()) for s in tail)
        finite=all(np.isfinite(s['cap_pose']).all() for s in samples)
        passed=finite and peak<5.
        if name=='axial_pull':
            retained=max(abs(s['axial_displacement_m']) for s in samples if s['time_s']<1)<.003
            passed=passed and retained and separated
        else:
            retained=None;passed=passed and speed<.03
        return {'case':name,'fixed':fixed,'passed':bool(passed),'peak_speed_m_s':peak,
                'last_second_max_speed_m_s':speed,'retained_at_003n':retained,
                'separated':separated,'max_applied_force_n':max_force,
                'override':override,'samples':samples}
    finally:
        world.stop();world.clear();World.clear_instance()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--hz',type=int,default=1920)
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--recorded-state',type=Path,
                   help='JSON with body/cap pose and velocities; shoes reconstructed at rest anchors, robot omitted')
    p.add_argument('--cases',nargs='+',default=['rest','centre_drop','rim_drop','corner_drop','axial_pull'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.hz <= 0 or a.hz % 60 or a.seconds < 2:
        p.error('Use a positive multiple of 60 Hz and at least 2 seconds')
    recorded=json.loads(a.recorded_state.read_text()) if a.recorded_state else None
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':640,'height':480,'anti_aliasing':0})
    results=[]
    try:
        for name in a.cases:
            result=run_case(name,not a.baseline,a.hz,a.seconds,recorded);results.append(result)
            print('MARKER_FIT',name,{k:v for k,v in result.items() if k not in ('override','samples')},flush=True)
            (a.output/'result.json').write_text(json.dumps({'gpt_called':False,'hz':a.hz,
                'scope':'Passive drops and guided axial fixture; not autonomous task success or exact failed-run replay',
                'passed':all(x['passed'] for x in results),'cases':results},indent=2)+'\n')
        return 0 if all(x['passed'] for x in results) else 2
    finally:app.close()


if __name__=='__main__':raise SystemExit(main())
