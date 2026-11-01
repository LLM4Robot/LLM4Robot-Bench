"""L2-02 image-fit layout with a fixed bent lower book and matching collision.

The lower book remains one dynamic rigid body. Its fixed bent shape represents
visible paper flex at reset; this does not claim deformable-page simulation.
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from robot_tasks.robots.transforms import pose_transform, transform_pose


def apply(stage, row, reference):
    from l2_video_placement import world, set_world
    box_path = row['l2_support_measurement']['fixture']
    box = pose_transform(reference['box_pose'])
    set_world(stage, box_path, box)
    top = pose_transform(reference['upper_cover_world_pose'])
    upper = top.copy()
    upper[:3, 3] = (top @ np.array([0., 0., -.00979, 1.]))[:3]
    offsets = reference['book_offsets_from_upper_m']
    for name, offset in offsets.items():
        placed = upper.copy()
        placed[:3, 3] = (upper @ np.array([*offset, 1.]))[:3]
        set_world(stage, row['tracked'][name]['path'], placed)
    lower_path = row['tracked']['lower_book']['path']
    lower = world(stage, lower_path)
    relative = np.linalg.inv(box) @ lower
    sag = reference.get('lower_book_interior_sag_m', 0.)
    clearance = reference['rim_clearance_m']
    inset = np.array([.08052632, .0815])
    blend = .025

    def bottom_z(xy):
        xy = np.asarray(xy)
        # The bend is authored relative to the physical box: the underside
        # rises continuously onto its rear/right rim and sags over the recess.
        # Small XY movement from the tilted local Z is included iteratively.
        z = np.full(xy.shape[:-1], -.01)
        for _ in range(5):
            p = np.concatenate([xy, z[..., None]], axis=-1)
            q = p @ relative[:3, :3].T + relative[:3, 3]
            d = np.clip((inset - np.abs(q[..., :2])) / blend, 0., 1.)
            smooth = d*d*(3. - 2.*d)
            if reference.get('lower_book_bend_profile') == 'continuous_lateral_arc':
                centre = reference['lower_book_sag_centre_box_x_m']
                x = q[..., 0] - centre
                span = np.where(x < 0., inset[0] + centre, inset[0] - centre)
                smooth[..., 0] = np.cos(np.pi * np.clip(x / span, -1., 1.) / 2.) ** 2
            if reference.get('lower_book_bend_profile') == 'straight_rising_front':
                # Keep the visible front edge straight and gently rising to
                # the right rim. Only the rear underside blends up to its rim;
                # a 22 cm rigid book cannot sink through the 20 cm box walls.
                angle = np.deg2rad(reference['lower_book_front_slope_deg'])
                plane = .070 + clearance + np.tan(angle) * (q[..., 0] - inset[0])
                plane = np.maximum(plane, .060 + clearance)
                height = plane + np.maximum(.070 + clearance - plane, 0.) * (1. - smooth[..., 1])
            else:
                height = .070 + clearance - sag * smooth[..., 0] * smooth[..., 1]
            z = (height - relative[2, 3] - xy @ relative[2, :2]) / relative[2, 2]
        return z

    changed = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(lower_path + '/Visuals')):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = np.linalg.inv(lower) @ world(stage, str(prim.GetPath()))
        inverse = np.linalg.inv(transform)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        p = points @ transform[:3, :3].T + transform[:3, 3]
        if prim.GetName() == 'BackCover' and 'lower_back_cover_half_depth_m' in reference:
            # The delivered cover projects ~1 mm past the page edges. Bending
            # made that lip look like a separate sheet; retain a small cover
            # margin while bringing its front/back edges close to the pages.
            p[:, 1] *= reference['lower_back_cover_half_depth_m'] / .110
        bottom = bottom_z(p[:, :2])
        weight = np.clip((.010 - p[:, 2]) / .020, 0., 1.)
        p[:, 2] += weight * (bottom + .010)
        p = p @ inverse[:3, :3].T + inverse[:3, 3]
        mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in p])
        mesh.GetExtentAttr().Set([Gf.Vec3f(*p.min(0)), Gf.Vec3f(*p.max(0))])
        # Let USD derive normals from the modified visible surface.
        prim.RemoveProperty('normals')
        prim.RemoveProperty('primvars:normals')
        changed.append(str(prim.GetPath()))
    envelope = stage.GetPrimAtPath(lower_path + '/Colliders/Envelope')
    material = envelope.GetRelationship('material:binding:physics').GetTargets()
    if not material:
        material = envelope.GetRelationship('material:binding').GetTargets()
    stage.RemovePrim(str(envelope.GetPath()))
    collision_root = lower_path + '/Colliders/BentEnvelope'
    xs, ys = np.linspace(-.07, .07, 15), np.linspace(-.11, .11, 23)
    count = 0
    for i, (x0, x1) in enumerate(zip(xs, xs[1:])):
        for j, (y0, y1) in enumerate(zip(ys, ys[1:])):
            xy = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            lo = np.column_stack((xy, bottom_z(xy)))
            hi = np.column_stack((xy, np.full(4, .010)))
            points = np.concatenate((lo, hi))
            mesh = UsdGeom.Mesh.Define(stage, f'{collision_root}/Cell_{i:02}_{j:02}')
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            mesh.CreateFaceVertexCountsAttr([4]*6)
            mesh.CreateFaceVertexIndicesAttr([3,2,1,0,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
            mesh.CreateVisibilityAttr('invisible')
            mesh.CreatePurposeAttr('guide')
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0002)
            prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
            if material:
                prim.CreateRelationship('material:binding:physics').SetTargets(material)
            count += 1
    # Bake the lower book's slope into its shape. Keep its body Z axis vertical
    # so the existing measured support AABB does not invent low corners outside
    # the bent envelope. Both visuals and collisions retain their world poses.
    from scipy.spatial.transform import Rotation
    upright = lower.copy()
    yaw = np.arctan2(lower[1, 0], lower[0, 0])
    upright[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    children = {name: world(stage, lower_path + '/' + name) for name in ('Visuals', 'Colliders')}
    set_world(stage, lower_path, upright)
    for name, matrix in children.items():
        set_world(stage, lower_path + '/' + name, matrix)
    # Store a surface definition for geometric verification, not just an AABB.
    return {'method': reference['fit_method'], 'limitation': reference['limitation'],
            'box_pose': reference['box_pose'], 'book_offsets_from_upper_m': offsets,
            'upper_cover_world_pose': reference['upper_cover_world_pose'],
            'lower_book_interior_sag_m': reference.get('lower_book_interior_sag_m'),
            'lower_book_front_slope_deg': reference.get('lower_book_front_slope_deg'),
            'lower_book_bend_profile': reference.get('lower_book_bend_profile', 'smoothstep'),
            'lower_back_cover_half_depth_m': reference.get('lower_back_cover_half_depth_m', .110),
            'upper_book_tilt_deg': float(np.rad2deg(np.arccos(top[2,2]))),
            'lower_book_fixed_bent_shape': True, 'all_three_books_remain_dynamic': True,
            'changed_visible_meshes': changed, 'matching_collision_cells': count,
            'new_hidden_supports_or_constraints': False,
            'support': 'Real raised rear/right box rim; the fixed lower underside clears the inset platform.'}
