"""Apply the user's measured support dimensions to L2 runtime layers only.

The original USDZ, materials, rounded rim topology and physical mechanisms stay
intact. The centre and rim use separate height mappings: uniform Z scaling would
not reproduce the measured 10 mm recess. Safe to apply again to the same layer.
"""

import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

TASKS = ('L2-01', 'L2-02', 'L2-04', 'L2-05')
REVISION = 'l2-user-measured-box-200x200x70-rim10-v1'
OLD_FLOOR, OLD_TOP = .070125, .075
FLOOR, TOP = .060, .070
XY_SCALE = np.array([.200 / .205, .200 / .195])
STAMP = 'gptBenchmark:l2MeasuredSupport'
GPU_STAMP = 'gptBenchmark:l2SupportGpuCollisions'


def gpu_collision_representation(stage, body):
    """Retain the exterior while avoiding GPU-incompatible thin, long hulls."""
    from scipy.spatial import ConvexHull
    stamp = body.GetAttribute(GPU_STAMP)
    if stamp and stamp.Get():
        return json.loads(stamp.Get())

    def clipped(points, axis, bound, less):
        hull = ConvexHull(points)
        edges = {tuple(sorted((a, b))) for face in hull.simplices
                 for a, b in zip(face, np.roll(face, 1))}
        result = [p for p in points if (p[axis] <= bound + 1e-10 if less else p[axis] >= bound - 1e-10)]
        for a, b in edges:
            p, q = points[a], points[b]
            if (p[axis] - bound) * (q[axis] - bound) < 0:
                result.append(p + (q - p) * (bound - p[axis]) / (q[axis] - p[axis]))
        return np.unique(np.asarray(result), axis=0)

    partitioned = {}
    root = str(body.GetPath())
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root + '/Colliders'))):
        if not prim.GetName().startswith('Rim_'):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        axis = int(np.argmax(np.ptp(points[:, :2], axis=0)))
        low, high = points[:, axis].min(), points[:, axis].max()
        if high - low < .05:
            continue
        if not np.allclose(UsdGeom.Xformable(prim).GetLocalTransformation(), np.eye(4), atol=1e-10):
            raise ValueError('Unexpected rim collider transform')
        template = Sdf.Layer.CreateAnonymous()
        Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), template, '/Prototype')
        total_volume = 0.
        for i in range(8):
            vertices = clipped(clipped(points, axis, low + (high - low) * i / 8, False),
                               axis, low + (high - low) * (i + 1) / 8, True)
            hull = ConvexHull(vertices)
            # Keep outward winding instead of relying on the hull cooker to fix it.
            faces = hull.simplices.copy()
            for f, normal in zip(faces, hull.equations[:, :3]):
                if np.dot(np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]]), normal) < 0:
                    f[1], f[2] = f[2], f[1]
            child_path = str(prim.GetPath()) + '/Segment' + str(i)
            Sdf.CopySpec(template, '/Prototype', stage.GetRootLayer(), child_path)
            child = UsdGeom.Mesh(stage.GetPrimAtPath(child_path))
            child.GetPointsAttr().Set(vertices.tolist())
            child.GetFaceVertexCountsAttr().Set([3] * len(faces))
            child.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())
            child.CreateExtentAttr().Set([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
            total_volume += hull.volume
        ratio = total_volume / ConvexHull(points).volume
        if not abs(ratio - 1.) < 1e-6:
            raise ValueError('Rim partition changes collision volume')
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        partitioned[prim.GetName()] = {'segments': 8, 'volume_ratio': ratio}
    # The inset's thin convex slab failed GPU cooking. Extend its bottom only
    # into the already solid LowerBody; the exposed surface stays at 60 mm.
    inset = UsdGeom.Mesh(stage.GetPrimAtPath(root + '/Colliders/InsetFace'))
    vertices = np.asarray(inset.GetPointsAttr().Get(), dtype=float)
    original_bottom = float(vertices[:, 2].min())
    vertices[vertices[:, 2] < FLOOR - 1e-6, 2] = .03
    inset.GetPointsAttr().Set(vertices.tolist())
    inset.CreateExtentAttr().Set([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
    result = {'revision': 'gpu-compatible-support-v1', 'partitioned_rims': partitioned,
              'inset_bottom_before_m': original_bottom, 'inset_bottom_after_m': .03,
              'exposed_collision_surface_unchanged': True,
              'inset_extension_inside_existing_solid_lower_body': True}
    body.CreateAttribute(GPU_STAMP, Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(result, sort_keys=True))
    return result


def world_matrix(cache, prim):
    return np.asarray(cache.GetLocalToWorldTransform(prim)).T


def move_world_z(stage, path, delta):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError('Missing supported assembly: ' + path)
    cache = UsdGeom.XformCache()
    world = world_matrix(cache, prim)
    world[2, 3] += delta
    parent = world_matrix(cache, prim.GetParent())
    local = np.linalg.inv(parent) @ world
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))


def apply(stage, tid, row):
    if tid not in TASKS:
        return None
    path = next(r['path'] for r in row['fixtures']
                if r['path'].endswith(('/support_box', '/Pedestal')))
    body = stage.GetPrimAtPath(path)
    stamp = body.GetAttribute(STAMP)
    if stamp and stamp.Get():
        evidence = json.loads(stamp.Get())
        if evidence['revision'] != REVISION:
            raise ValueError('Rebuild from the source delivery before changing measurement revision')
        if 'gpu_collision_representation' not in evidence:
            evidence['gpu_collision_representation'] = gpu_collision_representation(stage, body)
            stamp.Set(json.dumps(evidence, sort_keys=True))
        return evidence
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(world_matrix(cache, body))
    meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh)]
    converted = []
    for mesh in meshes:
        relative = inverse @ world_matrix(cache, mesh.GetPrim())
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        vertices = points @ relative[:3, :3].T + relative[:3, 3]
        converted.append((mesh, relative, vertices))
    bounds = np.concatenate([v for _, _, v in converted])
    if not np.allclose(np.ptp(bounds, axis=0), [.205, .195, OLD_TOP], atol=1e-6):
        raise ValueError('Unexpected delivered support geometry; refusing cumulative scaling')
    for mesh, relative, vertices in converted:
        below = vertices[:, 2] <= OLD_FLOOR + 1e-8
        slope = np.where(below, FLOOR / OLD_FLOOR, (TOP - FLOOR) / (OLD_TOP - OLD_FLOOR))
        vertices[:, :2] *= XY_SCALE
        vertices[:, 2] = np.where(below, vertices[:, 2] * slope,
                                 FLOOR + (vertices[:, 2] - OLD_FLOOR) * slope)
        points = (vertices - relative[:3, 3]) @ np.linalg.inv(relative[:3, :3]).T
        mesh.GetPointsAttr().Set(points.tolist())
        mesh.CreateExtentAttr().Set([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        if mesh.GetNormalsAttr().HasAuthoredValueOpinion():
            if mesh.GetNormalsInterpolation() != UsdGeom.Tokens.vertex:
                raise ValueError('Unexpected support normal interpolation: ' + str(mesh.GetPath()))
            normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=float)
            normals = normals @ np.linalg.inv(relative[:3, :3])
            normals /= np.column_stack((np.tile(XY_SCALE, (len(points), 1)), slope))
            normals = normals @ relative[:3, :3]
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            mesh.GetNormalsAttr().Set(normals.tolist())
    # Move whole assemblies, including their joints, so supported items retain
    # their original contact clearance. Only episode initialization changes.
    if tid == 'L2-01':
        shifts = {'/World/L2/marker': FLOOR - OLD_FLOOR}
    elif tid == 'L2-02':
        shifts = {r['path']: TOP - OLD_TOP for r in row['tracked'].values()}
    elif tid == 'L2-04':
        shifts = {row['orange']['path']: FLOOR - OLD_FLOOR}
    else:
        shifts = {'/World/L2/organizer': TOP - OLD_TOP}
    for prim_path, delta in shifts.items():
        move_world_z(stage, prim_path, delta)
    evidence = {
        'revision': REVISION, 'fixture': path,
        'source': 'User physical measurement, confirmed total height includes rim, 2026-10-15',
        'outer_dimensions_m': [.200, .200, TOP],
        'inset_platform_z_m': FLOOR, 'rim_rise_m': TOP - FLOOR,
        'inset_half_extents_m': [.082539 * XY_SCALE[0], .079462 * XY_SCALE[1]],
        'original_outer_dimensions_m': [.205, .195, OLD_TOP],
        'original_inset_platform_z_m': OLD_FLOOR,
        'meshes_updated': len(meshes), 'initial_world_z_shifts_m': shifts,
        'visible_materials_uv_topology_preserved': True,
        'visible_and_collision_geometry_updated': True,
        'source_usdz_modified': False,
    }
    evidence['gpu_collision_representation'] = gpu_collision_representation(stage, body)
    body.CreateAttribute(STAMP, Sdf.ValueTypeNames.String, custom=True).Set(json.dumps(evidence, sort_keys=True))
    return evidence


def refresh_metadata(stage, row):
    from video_scene_metadata import record
    row['tracked'] = {name: record(stage, r['path']) for name, r in row['tracked'].items()}
    row['fixtures'] = [record(stage, r['path']) for r in row['fixtures']]
    if 'orange' in row:
        row['orange'] = record(stage, row['orange']['path'])


def update_runtime_files():
    """Update the four existing runtime layers; preserve all other task records."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / 'assets/tuojing/video_20261015.json'
    setup_path = root / 'configs/task_references/video_visual_setup_20261015.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    setup_bytes = setup_path.read_bytes()
    setup = json.loads(setup_bytes)
    history = setup_path.parent / 'video_visual_setup_history' / (hashlib.sha256(setup_bytes).hexdigest() + '.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_bytes(setup_bytes)
    for tid in TASKS:
        row = manifest['tasks'][tid]
        path = root / 'assets/tuojing' / row['runtime_path']
        stage = Usd.Stage.Open(str(path))
        row['l2_support_measurement'] = apply(stage, tid, row)
        refresh_metadata(stage, row)
        row['physical_repairs']['support']['geometry_and_materials'] = 'Latest complete delivery; L2 dimensions corrected to user measurement, materials preserved'
        stage.GetRootLayer().Save()
        row['runtime_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        row['runtime_bytes'] = path.stat().st_size
        setup['tasks'][tid]['runtime_scene_sha256'] = row['runtime_sha256']
        print(tid, row['runtime_sha256'], flush=True)
    setup['revision'] = 'video_visible_setup_v20_l2_measured_box_20261015'
    for path, data in ((manifest_path, manifest), (setup_path, setup)):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)


if __name__ == '__main__':
    update_runtime_files()
