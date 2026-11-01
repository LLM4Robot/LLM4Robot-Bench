"""Restore task-specific glass-ball counts using the current delivered asset."""

from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
from robot_tasks.robots.transforms import pose_transform
from video_scene_metadata import record


def apply(stage, tid, rows, assets_root):
    if tid not in ('L1-05','L2-03'):
        return None
    source = rows['L2-03']
    package = assets_root/source['path']
    original = Usd.Stage.Open(str(package))
    source_cup = pose_transform(source['tracked']['source_cup']['pose'])
    inverse = np.linalg.inv(source_cup)
    target_name = 'target_cup' if tid == 'L1-05' else 'source_cup'
    target_cup = pose_transform(rows[tid]['tracked'][target_name]['pose'])
    root = '/World/Household' if tid == 'L1-05' else '/World/L2'
    cache = UsdGeom.XformCache()
    result = {}
    positions=[]
    count = 12 if tid == 'L2-03' else 10
    for i in range(count):
        template = source['tracked']['load_'+str(i%5)]
        relative = inverse @ pose_transform(template['pose'])
        if i >= 5:
            # The tapered cup cannot accommodate a ring at the bottom. Complete
            # the existing upper ring, then add the remaining balls above it.
            z=(inverse @ pose_transform(source['tracked']['load_1']['pose']))[2,3]
            if i < 8:
                angle=(i-6)*np.pi/3
                relative[:3,3]=[.0165*np.cos(angle),.0165*np.sin(angle),z]
            elif i < 10:
                relative[:3,3]=[(i-8.5)*.0165,0.,z+.0165]
            else:
                relative[:3,3]=[0.,(i-10.5)*.0165*np.sqrt(3),z+.0165]
        positions.append(relative[:3,3].copy())
        if tid == 'L2-03' and i < 5:
            continue  # Keep every original native initial pose.
        world = target_cup @ relative
        name = ('filler_' if tid == 'L1-05' else 'load_')+str(i)
        prim = stage.DefinePrim(root+'/'+name,'Xform')
        prim.GetReferences().AddReference(str(package.resolve()),Sdf.Path(template['path']))
        parent = np.asarray(cache.GetParentToWorldTransform(prim)).T
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d((np.linalg.inv(parent)@world).T.tolist()))
        result[name] = record(stage,str(prim.GetPath()))
    if tid == 'L2-03':
        # Contact reporting disables sleeping. Resolve the loaded cup's contact
        # stack accurately instead of freezing the cup or any glass sphere.
        bodies = {**rows[tid]['tracked'], **result}
        for name, body in bodies.items():
            if name in ('source_cup', 'receiver_cup') or name.startswith('load_'):
                prim = stage.GetPrimAtPath(body['path'])
                prim.GetAttribute('physxRigidBody:solverPositionIterationCount').Set(64)
                prim.GetAttribute('physxRigidBody:solverVelocityIterationCount').Set(16)
    from robot_tasks.tasks.bench.native_geometry import cup_cavity
    walls,floor,_=cup_cavity('video:ripple_paper_cup')
    positions=np.asarray(positions)
    clearances=(walls[:,3,None]-walls[:,:3]@positions.T)/np.linalg.norm(walls[:,:3],axis=1)[:,None]-.008
    distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    if clearances.min()<.0002 or positions[:,2].min()-.008-floor<.0002 or distances.min()<.0164:
        raise ValueError('Glass-ball packing penetrates a native cup wall, floor or another ball')
    metadata = {'added':result,'container':target_name,'count':count,'radius_m':.008,
            'packing_validation':{'minimum_wall_clearance_m':float(clearances.min()),
                'minimum_floor_clearance_m':float(positions[:,2].min()-.008-floor),
                'minimum_centre_distance_m':float(distances.min()),'physics_settling_required':True},
            'source_task':'L2-03','source_path':source['path'],
            'source_sha256':source['sha256'],'source_bytes':source['bytes'],
            'evidence':('User confirmed 12 glass balls for L2-03, 2026-10-15.' if tid == 'L2-03' else 'PR #19 L1 real-alignment record confirms ten glass balls; L2-03 real video inspection shows at least ten. Native saved scenes omitted these counts.'),
            'limits':'Native 16 mm size/mass retained; real size and hidden packing are not measured. No white pearls are invented.'}
    if tid == 'L2-03':
        metadata['contact_solver'] = {'position_iterations':64, 'velocity_iterations':16,
                                      'mass_friction_restitution_unchanged':True,
                                      'source_cup_and_all_balls_remain_dynamic':True}
    return metadata
