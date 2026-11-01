"""Rebuild the flower head contact shell from its visible outer petals.

Keeps the original visual model, stem, leaves, mass, poses and cup contents.
Only the inaccessible petal interior is filled by this rigid grasp proxy.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1]
FLOWER = '/World/Household/rose'


def apply(stage):
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(FLOWER))).T)
    points = []
    for prim in stage.GetPrimAtPath(FLOWER+'/Visuals').GetChildren():
        if prim.GetName().startswith('Petal') and prim.IsA(UsdGeom.Mesh):
            transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            points.extend(vertices @ transform[:3,:3].T + transform[:3,3])
    points = np.unique(np.asarray(points), axis=0)
    full_points = points.copy()
    # Keep the authored hull comfortably below PhysX's 255-vertex limit.
    # Farthest-point sampling retains all sides instead of relying on cooking
    # to silently simplify thousands of tiny petal triangles.
    chosen = [int(np.argmax(points[:,2]))]
    distance = np.full(len(points), np.inf)
    for _ in range(63):
        distance = np.minimum(distance, np.sum((points-points[chosen[-1]])**2,axis=1))
        chosen.append(int(np.argmax(distance)))
    for _ in range(176):
        hull = ConvexHull(full_points[chosen])
        errors = np.max(full_points@hull.equations[:,:3].T+hull.equations[:,3],axis=1)
        if errors.max() <= .0003:
            break
        chosen.append(int(np.argmax(errors)))
    points = full_points[chosen]
    hull = ConvexHull(points)
    approximation_error = float(np.max(full_points@hull.equations[:,:3].T+hull.equations[:,3]))
    if approximation_error > .0006:
        raise ValueError('Flower-head proxy differs from visible petals by over 0.6 mm')
    # Outward triangle winding, compact vertex list, no geometric inflation.
    faces = hull.simplices.copy()
    normals = np.cross(points[faces[:,1]]-points[faces[:,0]], points[faces[:,2]]-points[faces[:,0]])
    reverse = np.sum(normals*hull.equations[:,:3],axis=1) < 0
    faces[reverse] = faces[reverse, ::-1]
    chosen = sorted(set(faces.ravel())); mapping = {v:i for i,v in enumerate(chosen)}
    faces = np.array([[mapping[i] for i in f] for f in faces])
    points = points[chosen]
    for prim in list(stage.GetPrimAtPath(FLOWER+'/Colliders').GetChildren()):
        if prim.GetName().startswith('Petal'):
            stage.RemovePrim(prim.GetPath())
    mesh = UsdGeom.Mesh.Define(stage, FLOWER+'/Colliders/PetalHeadHull')
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateVisibilityAttr('invisible')
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    prim.AddAppliedSchema('PhysxCollisionAPI')
    prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0002)
    prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/World/Household/Material')), materialPurpose='physics')
    grasp = UsdGeom.Xform.Define(stage, FLOWER+'/Frames/HeadGrasp')
    point = (points.min(axis=0)+points.max(axis=0))/2
    matrix = Gf.Matrix4d(1.); matrix.SetTranslate(Gf.Vec3d(*point))
    grasp.MakeMatrixXform().Set(matrix)
    return {'revision':'outer_petal_grasp_hull_20261016_v1',
            'head_grasp_local_m':point.tolist(), 'head_hull_vertices':len(points),
            'max_petal_plane_error_m':approximation_error,
            'visuals_stem_leaves_mass_and_initial_poses_unchanged':True,
            'limitation':'Rigid convex outer flower-head proxy; internal petal cavity is filled, no soft petal deformation.'}


if __name__ == '__main__':
    manifest_path = ROOT/'assets/tuojing/video_20261015.json'
    manifest = json.loads(manifest_path.read_text()); row = manifest['tasks']['L1-05']
    path = ROOT/'assets/tuojing'/row['runtime_path']
    stage = Usd.Stage.Open(str(path)); repair = apply(stage)
    stage.GetRootLayer().Save()
    row['flower_collision_repair'] = repair
    row['runtime_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    row['runtime_bytes'] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    setup_path = ROOT/'configs/task_references/video_visual_setup_20261015.json'
    contents = setup_path.read_bytes()
    history = setup_path.parent/'video_visual_setup_history'; history.mkdir(exist_ok=True)
    (history/(hashlib.sha256(contents).hexdigest()+'.json')).write_bytes(contents)
    setup = json.loads(contents)
    setup['tasks']['L1-05']['runtime_scene_sha256'] = row['runtime_sha256']
    setup_path.write_text(json.dumps(setup,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(repair,ensure_ascii=False,indent=2))
