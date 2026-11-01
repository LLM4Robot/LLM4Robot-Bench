"""Passive L3-01 closure repair, shared by original and variant placement.

Keep the supplied visual meshes, masses and friction. Replace the two large
SDFs with hollow convex shell pieces and short segments of the supplied thread
meshes. A cap starts one thread pitch further engaged than the nearly released
delivery. There are no attachment joints, prescribed motions or applied forces.
"""

from copy import deepcopy
import math

import numpy as np

from ...robots.transforms import pose_transform, transform_pose

REVISION = "l301-passive-closure-v1"
THREAD_LEAD_M = 0.009
REENGAGE_M = 0.003
CONTACT_OFFSET_M = 0.00015


def initial_record(source):
    """Return an isolated, frame-consistent reset record; never mutate assets."""
    row = deepcopy(source)
    body = pose_transform(row['tracked']['bottle']['pose'])
    cap = pose_transform(row['tracked']['cap']['pose'])
    relative = np.linalg.inv(body) @ cap
    source_angle = math.radians(-60)
    source_rotation = pose_transform((0., 0., 0., math.cos(source_angle / 2),
                                      0., 0., math.sin(source_angle / 2)))[:3, :3]
    if (not np.allclose(relative[:3, 3], (0., 0., .0075), atol=1e-6)
            or not np.allclose(relative[:3, :3], source_rotation, atol=1e-6)):
        raise ValueError("Audit the changed bottle closure before applying its repair")
    angle = -2 * math.pi * REENGAGE_M / THREAD_LEAD_M
    delta = pose_transform((0., 0., -REENGAGE_M,
                            math.cos(angle / 2), 0., 0., math.sin(angle / 2)))
    world = body @ delta @ relative
    row['tracked']['cap']['pose'] = list(transform_pose(world))
    row['tracked']['cap']['native_world_matrix'] = world.tolist()
    row.setdefault('physical_repairs', {})['bottle_closure'] = {
        'revision': REVISION, 'source_modified': False,
        'initial_cap_lift_m': float(relative[2, 3] - REENGAGE_M),
        'initial_cap_adjustment_m': -REENGAGE_M,
        'initial_cap_adjustment_deg': math.degrees(angle),
        'thread_lead_m': THREAD_LEAD_M,
        'contact_model': 'hollow convex shell and segmented native threads',
        'opening_torque_calibrated': False,
    }
    return row


def apply_initial_pose(stage, row):
    from pxr import Gf, UsdGeom
    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError("Bottle repair must only edit the session layer")
    record = row['tracked']['cap']
    prim = stage.GetPrimAtPath(record['path'])
    parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    local = np.linalg.inv(parent) @ pose_transform(record['pose'])
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))


def ring_pieces(profiles, segments=32):
    """Convex annular sectors; split axial elbows instead of filling the cavity."""
    for section in range(len(profiles) - 1):
        for i in range(segments):
            angles = np.linspace(i * 2 * np.pi / segments, (i + 1) * 2 * np.pi / segments, 3)
            yield np.asarray([(r * np.cos(t), r * np.sin(t), z)
                              for z, inner, outer in profiles[section:section + 2]
                              for r in (inner, outer) for t in angles])


def configure(stage, root):
    """Author fixed collision geometry once, before PhysX/reset initialization."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial import ConvexHull

    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError("Bottle collision repair must only edit the session layer")
    material = UsdShade.Material(stage.GetPrimAtPath(root + '/Looks/Contact'))
    if not material:
        raise ValueError("Missing delivered bottle contact material")
    count = {'Body': 0, 'Cap': 0}

    def attr(prim, name, kind, value):
        prim.CreateAttribute(name, kind).Set(value)

    def hull(part, name, points):
        points = np.asarray(points, dtype=float)
        convex = ConvexHull(points)
        faces = convex.simplices.copy()
        normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                           points[faces[:, 2]] - points[faces[:, 0]])
        flip = np.sum(normals * convex.equations[:, :3], axis=1) < 0
        faces[flip] = faces[flip][:, ::-1]
        mesh = UsdGeom.Mesh.Define(stage, f'{root}/{part}/RobustContacts/{name}')
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateVisibilityAttr('invisible')
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        prim.AddAppliedSchema('PhysxCollisionAPI')
        attr(prim, 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, CONTACT_OFFSET_M)
        attr(prim, 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, 0.)
        prim.AddAppliedSchema('PhysxConvexHullCollisionAPI')
        attr(prim, 'physxConvexHullCollision:minThickness', Sdf.ValueTypeNames.Float, .00001)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        count[part] += 1

    for part, expected_threads in (('Body', 12), ('Cap', 3)):
        body = stage.GetPrimAtPath(root + '/' + part)
        sdf = stage.GetPrimAtPath(f'{root}/{part}/Colliders/{part}_SDF')
        threads = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(f'{root}/{part}/Visuals'))
                   if 'Thread_' in prim.GetName()]
        if not body or not sdf or len(threads) != expected_threads:
            raise ValueError("Unexpected bottle topology; audit before replacing its contacts")
        UsdPhysics.CollisionAPI(sdf).CreateCollisionEnabledAttr(False)
        body.AddAppliedSchema('PhysxRigidBodyAPI')
        attr(body, 'physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool, True)
        attr(body, 'physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, 64)
        attr(body, 'physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 8)
        for prim in threads:
            # Supplied helical meshes consist of four-vertex radial cross sections.
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).reshape(-1, 4, 3)
            for j in range(0, len(points) - 1, 8):
                hull(part, f'{prim.GetName()}_{j:03d}', points[j:j + 9].reshape(-1, 3))

    profiles = (
        ('Body', 'Neck', ((.133, .01815, .0188), (.138, .0167, .0175), (.1565, .0167, .0175))),
        ('Body', 'Support', ((.140, .0173, .0204), (.1415, .0173, .0204))),
        ('Body', 'Bead', ((.1422, .0174, .0189), (.1432, .0174, .0189))),
        ('Cap', 'Wall', ((.1442, .01925, .0205), (.1585, .01925, .0205))),
    )
    for part, name, profile in profiles:
        for i, points in enumerate(ring_pieces(profile)):
            hull(part, f'{name}_{i:03d}', points)
    hull('Cap', 'Roof', [(.0205 * np.cos(t), .0205 * np.sin(t), z)
                         for z in (.1585, .160)
                         for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)])
    return {'revision': REVISION, 'convex_shapes': count,
            'contact_offset_m': CONTACT_OFFSET_M, 'speculative_ccd': True,
            'thread_profiles': 'convex segments of supplied helical meshes',
            'visuals_mass_friction_unchanged': True, 'object_attachment': False,
            'opening_torque_calibrated': False}
