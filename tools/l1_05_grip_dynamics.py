"""Finite contact-patch friction and visual-shell mass properties for L1-05.

The flower stays one free rigid body, with its original 12 g total mass and
unchanged visual/collision surfaces. Parameters are engineering estimates,
not measured physical material or mass-distribution calibration.
"""
from pathlib import Path
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

ROOT = '/World/Household/rose'
REVISION = 'visual_shell_inertia_finite_grip_patch_v1'


def visual_mass_properties(stage):
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))).T)
    parts = []
    for prim in stage.GetPrimAtPath(ROOT+'/Visuals').GetChildren():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        schema = UsdGeom.Mesh(prim)
        points = np.asarray(schema.GetPointsAttr().Get(), dtype=float)
        transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = points @ transform[:3,:3].T + transform[:3,3]
        indices = np.asarray(schema.GetFaceVertexIndicesAttr().Get())
        faces, start = [], 0
        for size in schema.GetFaceVertexCountsAttr().Get():
            face = indices[start:start+size];start += size
            faces.extend([[face[0],face[i],face[i+1]] for i in range(1,size-1)])
        mesh = trimesh.Trimesh(vertices=points,faces=faces,process=True)
        if mesh.volume < 0:
            mesh.invert()
        if not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError('Cannot derive shell inertia from '+prim.GetName())
        props = mesh.mass_properties
        parts.append((prim.GetName(),float(props['volume']),np.asarray(props['center_mass']),np.asarray(props['inertia'])))
    mass = float(UsdPhysics.MassAPI(stage.GetPrimAtPath(ROOT)).GetMassAttr().Get())
    volume = sum(p[1] for p in parts)
    com = sum(p[1]*p[2] for p in parts)/volume
    inertia = np.zeros((3,3))
    for _,v,c,i in parts:
        d=c-com;inertia += i+v*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    inertia *= mass/volume
    values,axes=np.linalg.eigh(inertia)
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    q=Rotation.from_matrix(axes).as_quat()
    return mass,com,values,Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])),[
        {'part':n,'mass_kg':v*mass/volume} for n,v,_,_ in parts]


def apply(stage):
    mass,com,inertia,axes,parts=visual_mass_properties(stage)
    body=stage.GetPrimAtPath(ROOT);api=UsdPhysics.MassAPI(body)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    api.CreatePrincipalAxesAttr(axes)
    body.AddAppliedSchema('PhysxRigidBodyAPI')
    body.CreateAttribute('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float).Set(.3)
    mat=UsdShade.Material.Define(stage,ROOT+'/Looks/GripContact')
    physics=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(.8);physics.CreateDynamicFrictionAttr(.8);physics.CreateRestitutionAttr(0.)
    patches={}
    for name,radius in [('CoatedStem',.0019),('PetalHeadHull',.004)]:
        prim=stage.GetPrimAtPath(ROOT+'/Colliders/'+name)
        prim.AddAppliedSchema('PhysxCollisionAPI')
        for attr in ['torsionalPatchRadius','minTorsionalPatchRadius']:
            prim.CreateAttribute('physxCollision:'+attr,Sdf.ValueTypeNames.Float).Set(radius)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
        patches[name]=radius
    return {'revision':REVISION,'mass_kg':mass,'center_of_mass_local_m':com.tolist(),
            'principal_inertia_kg_m2':inertia.tolist(),'component_mass_estimates':parts,
            'mass_basis':'uniform-density visible thin shells, normalized to existing total mass; not solid flower-head collision hull',
            'torsional_patch_radius_m':patches,'static_friction':.8,'dynamic_friction':.8,
            'finger_patch_radius_m':.004,
            'finger_patch_basis':'estimated 8 mm axial contact span on the finger pad; not measured contact calibration',
            'visuals_and_collision_surfaces_unchanged':True,'free_rigid_body':True,
            'limitations':'Rigid flower; friction and homogeneous shell mass distribution are unmeasured engineering estimates.'}
