"""L1-05 contact approximations for light leaves and a coated-wire stem.

The original visual meshes remain; rounded vein proxies keep leaf contacts. No grip attachment,
pose following, gravity change or disabling of PhysX collision pairs is used. Spring
parameters approximate a flexible plastic leaf; they are not measurements.
"""
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = '/World/Household/rose'
REVISION = 'passive_leaves_bounded_deformation_v8'
LEAF_CONTACT_STIFFNESS = 2.  # N/m per contact point on a thin flexible leaf
LEAF_CONTACT_DAMPING = .02  # N s/m
LEAF_ROOT_STIFFNESS = .00025  # N m / degree
LEAF_ROOT_DAMPING = .000003  # N m / (degree / s)


def _properties(stage, prim, inverse):
    schema = UsdGeom.Mesh(prim)
    points = np.asarray(schema.GetPointsAttr().Get(), dtype=float)
    transform = inverse @ np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    vertices = points @ transform[:3, :3].T + transform[:3, 3]
    faces, offset = [], 0
    indices = np.asarray(schema.GetFaceVertexIndicesAttr().Get())
    for count in schema.GetFaceVertexCountsAttr().Get():
        polygon = indices[offset:offset+count]
        faces.extend([[polygon[0], polygon[i], polygon[i+1]] for i in range(1, count-1)])
        offset += count
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('Leaf mass needs a closed visual shell: ' + str(prim.GetPath()))
    return mesh.mass_properties, vertices


def _mass(prim, mass, com, tensor):
    values, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))


def _rounded_veins(stage, result, material):
    """A thin flexible blade is not a thick rigid hook around an open finger.

    Six rounded proxies follow each leaf's central vein. The wide blade stays
    visible but is not modeled as a rigid sheet. This is a collision
    approximation, not a soft-body reconstruction of the blade surface.
    """
    for record in result['leaf_bodies']:
        parent=stage.GetPrimAtPath(record['path']+'/Colliders')
        for prim in list(parent.GetChildren()):
            if prim.IsA(UsdGeom.Capsule):
                prim.CreateAttribute('task:compliantContactDeformationLimit',Sdf.ValueTypeNames.Float).Set(.008)
                prim.CreateAttribute('task:compliantContactForceLimit',Sdf.ValueTypeNames.Float).Set(.05)
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            low,high=np.percentile(points[:,2],[10,90])
            a=points[points[:,2]<=low].mean(axis=0)
            b=points[points[:,2]>=high].mean(axis=0)
            direction=b-a;length=np.linalg.norm(direction)
            if length<=.002:raise ValueError('Degenerate leaf vein proxy')
            rotation=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*(direction/length))).GetQuat()
            path=prim.GetPath();stage.RemovePrim(path)
            capsule=UsdGeom.Capsule.Define(stage,path)
            capsule.CreateAxisAttr('Z');capsule.CreateRadiusAttr(.0008)
            capsule.CreateHeightAttr(float(length));capsule.AddTranslateOp().Set(Gf.Vec3d(*((a+b)/2)))
            capsule.AddOrientOp().Set(Gf.Quatf(float(rotation.GetReal()),Gf.Vec3f(*rotation.GetImaginary())))
            capsule.CreateVisibilityAttr('invisible')
            capsule.GetPrim().CreateAttribute('task:compliantContactDeformationLimit',Sdf.ValueTypeNames.Float).Set(.008)
            capsule.GetPrim().CreateAttribute('task:compliantContactForceLimit',Sdf.ValueTypeNames.Float).Set(.05)
            collision=UsdPhysics.CollisionAPI.Apply(capsule.GetPrim());collision.CreateCollisionEnabledAttr(True)
            capsule.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
            for attr,value in [('contactOffset',.0003),('restOffset',0.),('torsionalPatchRadius',.0008)]:
                capsule.GetPrim().CreateAttribute('physxCollision:'+attr,Sdf.ValueTypeNames.Float).Set(value)
            UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(material,materialPurpose='physics')
    result.update(soft_contact_limits={'force_n':.05,'deformation_m':.008},
                  visual_meshes_and_collision_surfaces_preserved=False,
                  visual_meshes_preserved=True,
                  leaf_collision_representation='six rounded 0.8 mm radius central-vein capsules per leaf; flexible blade is visual only',
                  limitation='Passive leaf-root joints and rounded central-vein collision proxies approximate light flexible leaves; wide blades have no rigid-sheet collision. Not measured soft-body/FEM reconstruction. Stem and head remain rigid.')
    return result


def _stem_dissipation(stage, result):
    # The long rigid stem otherwise acts as an undamped pendulum and strikes
    # the cup rim. Viscous rotational damping approximates energy lost by a
    # real coated-wire stem bending. No target pose, force assist or anchoring.
    prim=stage.GetPrimAtPath(ROOT)
    prim.CreateAttribute('physxRigidBody:angularDamping',Sdf.ValueTypeNames.Float).Set(6.)
    result['stem_dissipation']={'angular_damping_per_s':6.,
        'model':'viscous rigid-proxy approximation of coated-stem bending losses',
        'calibration':'engineering approximation; not a measured flexible-beam material'}


def _container_friction(stage, result):
    # Keep the stem/finger pair grippy, while glass keeps its existing lower
    # friction rather than being averaged with the stem's rubber-like value.
    grip=stage.GetPrimAtPath(ROOT+'/Looks/GripContact')
    grip.AddAppliedSchema('PhysxMaterialAPI')
    grip.CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set('min')
    material=UsdShade.Material.Define(stage,'/World/Household/target_cup/Looks/InnerWallContact')
    physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.25);physics.CreateDynamicFrictionAttr(.18);physics.CreateRestitutionAttr(0.)
    material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set('min')
    for prim in stage.GetPrimAtPath('/World/Household/target_cup/Colliders').GetChildren():
        if prim.GetName().startswith('Wall_'):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
    result['container_friction']={'target_wall_static':.25,'target_wall_dynamic':.18,
        'stem_friction_combine':'min','stem_finger_friction_unchanged':True,
        'glass_mass_count_and_material_unchanged':True,
        'calibration':'unmeasured smooth coated-paper/glass engineering approximation'}


def apply(stage):
    root = stage.GetPrimAtPath(ROOT)
    material = UsdShade.Material.Define(stage, ROOT + '/Looks/FlexibleLeafContact')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.35)
    physics.CreateDynamicFrictionAttr(.25)
    physics.CreateRestitutionAttr(0.)
    material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    # A root hinge alone cannot yield at the first millimetres of the leaf.
    # Normal contact compliance approximates that local thin-plastic bending.
    # Keep actual collisions; unlike a rigid constraint, force grows gradually
    # with indentation and has viscous damping instead of an impact bounce.
    for attr, value in [('compliantContactStiffness', LEAF_CONTACT_STIFFNESS),
                        ('compliantContactDamping', LEAF_CONTACT_DAMPING)]:
        material.GetPrim().CreateAttribute('physxMaterial:' + attr, Sdf.ValueTypeNames.Float).Set(value)
    material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('min')
    existing = root.GetAttribute('task:passiveLeafAssembly')
    if existing and existing.Get():
        result = json.loads(existing.Get())
        result.update(revision=REVISION, leaf_contact_stiffness_N_per_m=LEAF_CONTACT_STIFFNESS,
                      leaf_contact_damping_Ns_per_m=LEAF_CONTACT_DAMPING)
        for record in result['leaf_bodies']:
            joint = stage.GetPrimAtPath(record['joint'])
            for axis in ['rotX', 'rotY', 'rotZ']:
                drive = UsdPhysics.DriveAPI(joint, axis)
                drive.GetStiffnessAttr().Set(LEAF_ROOT_STIFFNESS)
                drive.GetDampingAttr().Set(LEAF_ROOT_DAMPING)
        result.update(spring_stiffness_Nm_per_degree=LEAF_ROOT_STIFFNESS,
                      damping_Nm_per_degree_s=LEAF_ROOT_DAMPING)
        _rounded_veins(stage, result, material)
        _container_friction(stage, result)
        _stem_dissipation(stage, result)
        existing.Set(json.dumps(result))
        return result
    from l1_05_grip_dynamics import visual_mass_properties, apply as grip_dynamics
    total, _, _, _, parts = visual_mass_properties(stage)
    masses = {part['part']: part['mass_kg'] for part in parts}
    world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)).T
    inverse = np.linalg.inv(world)
    records = []
    for name in ['Leaf0', 'Leaf1']:
        original = stage.GetPrimAtPath(ROOT + '/Visuals/' + name)
        properties, points = _properties(stage, original, inverse)
        path = ROOT + '/LeafBodies/' + name
        body = UsdGeom.Xform.Define(stage, path)
        body.MakeMatrixXform().Set(Gf.Matrix4d(world.T.tolist()))
        body.SetResetXformStack(True)
        prim = body.GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        for attr, value in [('linearDamping', .03), ('angularDamping', .15),
                            ('maxDepenetrationVelocity', .1)]:
            prim.CreateAttribute('physxRigidBody:' + attr, Sdf.ValueTypeNames.Float).Set(value)
        prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(32)
        prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(8)
        mass = masses[name]
        _mass(prim, mass, properties['center_mass'], properties['inertia'] * mass / properties['volume'])
        UsdGeom.Xform.Define(stage, path + '/Visuals')
        UsdGeom.Xform.Define(stage, path + '/Colliders')
        layer = stage.GetRootLayer()
        Sdf.CopySpec(layer, original.GetPath(), layer, Sdf.Path(path + '/Visuals/' + name))
        stage.RemovePrim(original.GetPath())
        shapes = [p for p in stage.GetPrimAtPath(ROOT + '/Colliders').GetChildren()
                  if p.GetName().startswith(name + 'Contact')]
        for shape in shapes:
            target = path + '/Colliders/' + shape.GetName()
            Sdf.CopySpec(layer, shape.GetPath(), layer, Sdf.Path(target))
            stage.RemovePrim(shape.GetPath())
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(target)).Bind(material, materialPurpose='physics')
        # The lowest narrow section is where the authored leaf joins the stem.
        base = points[points[:, 2] <= points[:, 2].min() + .002].mean(axis=0)
        joint = UsdPhysics.Joint.Define(stage, ROOT + '/LeafJoints/' + name)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(ROOT)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*base))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*base))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        # Only the joined leaf/stem pair excludes self collision, as a normal
        # connected joint. Fingers, cameras, cups and other leaves still collide.
        joint.CreateCollisionEnabledAttr(False)
        for axis in ['transX', 'transY', 'transZ']:
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.)
            limit.CreateHighAttr(-1.)
        for axis, angle in [('rotX', 65.), ('rotY', 65.), ('rotZ', 30.)]:
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(-angle)
            limit.CreateHighAttr(angle)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr('force')
            drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            drive.CreateStiffnessAttr(LEAF_ROOT_STIFFNESS)  # N m / degree
            drive.CreateDampingAttr(LEAF_ROOT_DAMPING)  # N m / (degree / s)
            drive.CreateMaxForceAttr(.0025)   # N m; passive root bending torque
        records.append({'path': path, 'mass_kg': mass, 'joint': str(joint.GetPath()),
                        'anchor_local_m': base.tolist(), 'colliders': len(shapes)})
    UsdPhysics.MassAPI(root).GetMassAttr().Set(total - sum(r['mass_kg'] for r in records))
    root_dynamics = grip_dynamics(stage)
    result = {'revision': REVISION, 'leaf_bodies': records, 'total_flower_mass_kg': total,
              'root_mass_kg': root_dynamics['mass_kg'], 'root_grip_dynamics': root_dynamics,
              'spring_stiffness_Nm_per_degree': LEAF_ROOT_STIFFNESS, 'damping_Nm_per_degree_s': LEAF_ROOT_DAMPING,
              'max_passive_torque_Nm': .0025, 'leaf_friction': [.35, .25],
              'leaf_contact_stiffness_N_per_m': LEAF_CONTACT_STIFFNESS,
              'leaf_contact_damping_Ns_per_m': LEAF_CONTACT_DAMPING,
              'visual_meshes_and_collision_surfaces_preserved': True,
              'no_robot_attachment': True,
              'limitation': 'Two passive leaf-root joints approximate bending; not a measured soft-body/FEM reconstruction. Stem and flower head remain rigid.'}
    _rounded_veins(stage, result, material)
    _container_friction(stage, result)
    _stem_dissipation(stage, result)
    root.CreateAttribute('task:passiveLeafAssembly', Sdf.ValueTypeNames.String).Set(json.dumps(result))
    return result
