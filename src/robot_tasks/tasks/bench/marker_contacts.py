"""L2-01 contact geometry and passive cap retention.

The cap's 241 small hulls and the box's overlapping rim hulls can form
contradictory contact constraints after a release. For L2-01 use five solid
box pieces and separate cap envelopes for support and robot contact. The barrel
and spring shoes still use the hollow cap; fingers touch a continuous exterior.
"""

import numpy as np


def refresh_delivered_groups(stage, marker_path, *, simplify=True):
    """Keep a saved cap/box envelope filtered from all current robot geometry.

    A snapshot's explicit OtherColliders list predates added wrist detail, so
    rebuild memberships. The optional baseline is for diagnostics only.
    """
    from pxr import UsdPhysics
    repair = None
    support_repair = None
    if simplify:
        support_repair = configure_marker_box_contacts(stage, marker_path, '/World/L2/support_box')
        configure_marker_robot_contacts(stage, marker_path)
        from .marker_fit import simplify_delivered_marker
        repair = simplify_delivered_marker(stage, marker_path)
    root = marker_path+'/BoxContactGroups'
    names = {'Support':[], 'NativeCapAndShoes':[], 'CapEnvelope':[], 'OtherColliders':[],
             'RobotEnvelope':[], 'RobotColliders':[]}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        path = str(prim.GetPath())
        name = ('CapEnvelope' if path == marker_path+'/Cap/BoxContactEnvelope' else
                'RobotEnvelope' if path == marker_path+'/Cap/RobotContactEnvelope' else
                'RobotColliders' if simplify and path.startswith('/World/Robots/') else
                'NativeCapAndShoes' if path.startswith(marker_path+'/Cap/Colliders/') or path.startswith(marker_path+'/Shoe') else
                'Support' if path.startswith('/World/L2/support_box/') else 'OtherColliders')
        names[name].append(prim.GetPath())
    for name, paths in names.items():
        group = UsdPhysics.CollisionGroup.Define(stage, root+'/'+name)
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(paths)
    if simplify:
        UsdPhysics.CollisionGroup(stage.GetPrimAtPath(root+'/NativeCapAndShoes')).CreateFilteredGroupsRel().SetTargets(
            [root+'/Support', root+'/RobotColliders'])
        UsdPhysics.CollisionGroup(stage.GetPrimAtPath(root+'/RobotEnvelope')).CreateFilteredGroupsRel().SetTargets(
            [root+'/'+n for n in ('Support','NativeCapAndShoes','CapEnvelope','OtherColliders')])
        # Explicit shape filters supplement groups, including newly added wrist
        # geometry. Interior cap walls must not fight the exterior proxy.
        for path in names['NativeCapAndShoes']:
            shape = stage.GetPrimAtPath(path)
            pairs = UsdPhysics.FilteredPairsAPI.Apply(shape)
            previous = pairs.GetFilteredPairsRel().GetTargets()
            pairs.CreateFilteredPairsRel().SetTargets(list(dict.fromkeys(previous + names['RobotColliders'])))
        robot_proxy = stage.GetPrimAtPath(marker_path+'/Cap/RobotContactEnvelope')
        UsdPhysics.FilteredPairsAPI.Apply(robot_proxy).CreateFilteredPairsRel().SetTargets(
            [p for name, paths in names.items() if name not in ('RobotEnvelope','RobotColliders') for p in paths])
    envelope = stage.GetPrimAtPath(marker_path+'/Cap/BoxContactEnvelope')
    # Make the saved proxy's intended box-only scope explicit at the shape
    # level as well. It is solid and must never collide with the inserted barrel
    # or newly added robot shapes; native hollow-cap contacts stay enabled.
    UsdPhysics.FilteredPairsAPI.Apply(envelope).CreateFilteredPairsRel().SetTargets(
        names['OtherColliders']+names['NativeCapAndShoes']+names['RobotColliders']+names['RobotEnvelope'])
    return {**{name:len(paths) for name,paths in names.items()},
            'revision': 'l201-contact-roles-v3' if simplify else 'source-contact-groups',
            'envelope_scope':'source support solids only; explicit filtered pairs',
            'native_barrel_cap_fit_dimensions_preserved':True,
            'support_repair': support_repair,
            'robot_cap_contact': ('continuous outer cylinder; internal fit unchanged; robot cannot enter cap cavity'
                                  if simplify else 'delivered hollow wall hulls'),
            'contact_repair': repair}


def configure_marker_robot_contacts(stage, marker_path):
    """A single exterior for gripping, isolated from the inserted barrel.

The hollow 16-sector shell is required for cap/neck fit, but a jaw can contact
several thin sectors and their inner faces simultaneously. The external grip
uses the measured outer envelope, with the same mass, material and extent.
This deliberately omits jaw insertion into the cap mouth.
"""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    source = UsdGeom.Cylinder(stage.GetPrimAtPath(marker_path+'/Cap/BoxContactEnvelope'))
    proxy = UsdGeom.Cylinder.Define(stage, marker_path+'/Cap/RobotContactEnvelope')
    proxy.CreateAxisAttr(source.GetAxisAttr().Get())
    proxy.CreateRadiusAttr(source.GetRadiusAttr().Get())
    proxy.CreateHeightAttr(source.GetHeightAttr().Get())
    proxy.MakeMatrixXform().Set(source.GetLocalTransformation())
    proxy.CreateVisibilityAttr('invisible')
    shape = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(shape).CreateCollisionEnabledAttr(True)
    material = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial('physics')[0]
    UsdShade.MaterialBindingAPI.Apply(shape).Bind(material, materialPurpose='physics')
    shape.AddAppliedSchema('PhysxCollisionAPI')
    shape.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0002)
    shape.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)


def support_prisms(points, floor_z, top_z, half_opening):
    """Five convex prisms; preserve outer footprint, floor and rim heights.

    Only the decorative stepped grooves and inner corner fillets are omitted.
    Full-depth rim pieces have no thin lower faces that can trap another body
    against the support floor after numerical penetration.
    """
    points = sorted(set(map(tuple, np.asarray(points)[:, :2])))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def chain(items):
        result = []
        for point in items:
            while len(result) > 1 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    outline = np.asarray(chain(points)[:-1] + chain(points[::-1])[:-1])

    def clip(polygon, axis, bound, sign):
        output = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
            if da >= 0:
                output.append(a)
            if (da >= 0) != (db >= 0):
                output.append(a + (b - a) * da / (da - db))
        return np.asarray(output)

    hx, hy = half_opening
    centre_strip = clip(clip(outline, 0, -hx, 1), 0, hx, -1)
    return [("Floor", outline, floor_z),
            ("Left", clip(outline, 0, -hx, -1), top_z),
            ("Right", clip(outline, 0, hx, 1), top_z),
            ("Front", clip(centre_strip, 1, -hy, -1), top_z),
            ("Back", clip(centre_strip, 1, hy, 1), top_z)]


def configure_marker_box_contacts(stage, marker_path, box_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    box = stage.GetPrimAtPath(box_path)
    shell = stage.GetPrimAtPath(box_path + "/Visuals/SteppedMoldedShell")
    if not shell or not shell.IsA(UsdGeom.Mesh):
        raise ValueError("Marker support box requires its delivered closed shell")
    if not UsdPhysics.RigidBodyAPI(box).GetKinematicEnabledAttr().Get():
        raise ValueError("Marker support must be fixed/kinematic")

    cap = stage.GetPrimAtPath(marker_path + "/Cap")
    # Measure the delivered meshes, including disabled decorative top pieces
    # on a repeated call. Exclude the existing solid box-only envelope.
    native = [p for p in Usd.PrimRange(stage.GetPrimAtPath(marker_path + '/Cap/Colliders'))
              if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh)]
    if not native:
        raise ValueError("Unexpected native marker-cap collision geometry")
    transforms = UsdGeom.XformCache()
    cap_inverse = np.linalg.inv(np.asarray(transforms.GetLocalToWorldTransform(cap)).T)
    vertices = []
    for shape in native:
        matrix = cap_inverse @ np.asarray(transforms.GetLocalToWorldTransform(shape)).T
        points = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get())
        vertices.append(points @ matrix[:3, :3].T + matrix[:3, 3])
    vertices = np.concatenate(vertices)
    radius = float(np.linalg.norm(vertices[:, :2], axis=1).max())
    lower, upper = map(float, (vertices[:, 2].min(), vertices[:, 2].max()))
    if not 0 < radius < 0.02 or not 0 < upper - lower < 0.10:
        raise ValueError("Unexpected marker-cap size; audit the new asset first")

    # Preserve each delivered contact material, including friction/restitution.
    box_material = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath(box_path + "/Colliders/LowerBody")
    ).ComputeBoundMaterial("physics")[0]
    cap_material = UsdShade.MaterialBindingAPI(native[0]).ComputeBoundMaterial("physics")[0]
    if not box_material or not cap_material:
        raise ValueError("Missing native marker/box physics material")
    for shape in Usd.PrimRange(box):
        if shape.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(shape).CreateCollisionEnabledAttr(False)
    shell_points = np.asarray(UsdGeom.Mesh(shell).GetPointsAttr().Get())
    inset = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(
        box_path + "/Colliders/InsetFace")).GetPointsAttr().Get())
    floor_z, top_z = float(inset[:, 2].max()), float(shell_points[:, 2].max())
    half_opening = np.abs(inset[:, :2]).max(axis=0)
    box_shapes = []
    for name, outline, height in support_prisms(shell_points, floor_z, top_z, half_opening):
        n = len(outline)
        points = [(float(x), float(y), z) for z in (0., height) for x, y in outline]
        faces = []
        for i in range(1, n - 1):
            faces.extend([(0, i + 1, i), (n, n + i, n + i + 1)])
        for i in range(n):
            j = (i + 1) % n
            faces.extend([(i, j, n + j), (i, n + j, n + i)])
        mesh = UsdGeom.Mesh.Define(stage, box_path + "/ContactSolids/" + name)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateVisibilityAttr("invisible")
        shape = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(shape).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(shape).Bind(box_material, materialPurpose="physics")
        box_shapes.append(shape)

    envelope = UsdGeom.Cylinder.Define(stage, str(cap.GetPath()) + "/BoxContactEnvelope")
    envelope.CreateAxisAttr("Z")
    envelope.CreateRadiusAttr(radius)
    envelope.CreateHeightAttr(upper - lower)
    envelope.MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, (lower + upper) / 2)))
    envelope.CreateVisibilityAttr("invisible")
    proxy = envelope.GetPrim()
    UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(proxy).Bind(cap_material, materialPurpose="physics")
    for shape in (*box_shapes, proxy):
        shape.AddAppliedSchema('PhysxCollisionAPI')
        shape.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0002)
        shape.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)

    # Only the cap/box pair uses the envelope. In particular, it must NOT fill
    # the cap cavity against the barrel or change what the robot can grasp.
    shoe_shapes = [p for i in range(4)
                   for p in Usd.PrimRange(stage.GetPrimAtPath(f"{marker_path}/Shoe{i}"))
                   if p.HasAPI(UsdPhysics.CollisionAPI)]
    others = [p for p in stage.Traverse()
              if p.HasAPI(UsdPhysics.CollisionAPI)
              and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
              and p != proxy and p not in box_shapes + native + shoe_shapes]
    group_root = marker_path + "/BoxContactGroups"

    def group(name, shapes):
        result = UsdPhysics.CollisionGroup.Define(stage, group_root + "/" + name)
        result.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
            [p.GetPath() for p in shapes]
        )
        return result

    support = group("Support", box_shapes)
    fit = group("NativeCapAndShoes", native + shoe_shapes)
    exterior = group("CapEnvelope", [proxy])
    rest = group("OtherColliders", others)
    fit.CreateFilteredGroupsRel().SetTargets([support.GetPath()])
    exterior.CreateFilteredGroupsRel().SetTargets([rest.GetPath(), fit.GetPath()])
    return {
        "revision": "marker-box-contact-v1",
        "box_collision": "five full-depth convex solids, preserving delivered outer footprint, rim/floor heights and recess width",
        "box_floor_z_m": floor_z,
        "box_rim_z_m": top_z,
        "box_recess_half_extents_m": half_opening.tolist(),
        "cap_box_collision": "convex cylinder envelope; cap/box pair only",
        "cap_radius_m": radius,
        "cap_axial_bounds_m": [lower, upper],
        "native_cap_shapes_retained": len(native),
        "native_fit_dimensions_preserved": True,
        "box_envelope_excludes_robot": True,
        "native_mass_inertia_materials_and_springs_preserved": True,
        "source_assets_modified": False,
        "limitation": "Box decorative grooves/inner corner fillets are simplified. Box edges cannot enter the cap cavity/top recess; native barrel-cap fit is unchanged.",
    }
