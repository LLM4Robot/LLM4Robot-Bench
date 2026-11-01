"""Derive body-frame metadata from the delivered scene, including inherited scale."""

from itertools import product
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from robot_tasks.robots.transforms import transform_pose


def record(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError("Missing delivery prim: " + path)
    cache = UsdGeom.XformCache()
    world = np.asarray(cache.GetLocalToWorldTransform(prim)).T
    u, _, vt = np.linalg.svd(world[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    rigid = np.eye(4)
    rigid[:3, :3], rigid[:3, 3] = rotation, world[:3, 3]
    inverse = np.linalg.inv(rigid)
    points = []
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        image = UsdGeom.Imageable(shape)
        if not image or image.ComputeVisibility() == "invisible" or image.ComputePurpose() == "guide":
            continue
        if shape.IsA(UsdGeom.Mesh):
            vertices = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get())
        elif shape.IsA(UsdGeom.Gprim):
            b = bbox.ComputeUntransformedBound(shape).ComputeAlignedRange()
            vertices = np.array(list(product(*zip(b.GetMin(), b.GetMax()))))
        else:
            continue
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            continue
        relative = inverse @ np.asarray(cache.GetLocalToWorldTransform(shape)).T
        points.append(vertices @ relative[:3, :3].T + relative[:3, 3])
    if points:
        vertices = np.concatenate(points)
        bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    else:
        b = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
        bounds = [list(b.GetMin()), list(b.GetMax())]
    return {"path": path, "pose": list(transform_pose(rigid)), "bounds": bounds,
            "native_world_matrix": world.tolist(),
            "inherited_scale": np.linalg.norm(world[:3, :3], axis=0).tolist(),
            "mass": UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
            "kinematic": UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()}


def enrich(stage, task, tid):
    task['table'] = record(stage,'/World/Furniture/Worktable/Top')
    task['sdf_colliders'] = [str(p.GetPath()) for p in stage.Traverse()
                             if p.GetAttribute('physics:approximation')
                             and p.GetAttribute('physics:approximation').Get() == 'sdf']
    if tid == "L3-02" and "body" in task["tracked"]:
        task["tracked"]["anchor"] = task["tracked"].pop("body")
    for n, row in list(task["tracked"].items()):
        task["tracked"][n] = record(stage, row["path"])
    task["fixtures"] = [record(stage, row["path"]) for row in task["fixtures"]]
    if "orange" in task:
        task["orange"] = record(stage, task["orange"]["path"])
    for row in task["cameras"].values():
        cam = UsdGeom.Camera(stage.GetPrimAtPath(row["path"]))
        w, h = row["resolution"]
        ha, va = cam.GetHorizontalApertureAttr().Get(), cam.GetVerticalApertureAttr().Get()
        ox, oy = cam.GetHorizontalApertureOffsetAttr().Get(), cam.GetVerticalApertureOffsetAttr().Get()
        row["intrinsics"] = [w*cam.GetFocalLengthAttr().Get()/ha,
                             h*cam.GetFocalLengthAttr().Get()/va, w*(.5-ox/ha), h*(.5+oy/va)]
        row["aperture_offsets"] = [ox, oy]
    if tid == "L1-02":
        # Socket0's two openings measured in the delivered strip frame. These
        # values describe this hash-locked scene, not a hardware measurement.
        task["slot_centres"] = [[-.01, -.05735], [-.01, -.04365]]
        task["socket_frame_z"] = .02215
    if tid == "L3-03":
        cache = UsdGeom.XformCache()
        lines = []
        for i in range(4):
            prim = stage.GetPrimAtPath('/World/L2/Fixture/target_outline_'+str(i))
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            lines.append((matrix[:2,3], matrix[:2,0]))
        vertices = []
        for a,b in zip(lines, lines[1:]+lines[:1]):
            factors = np.linalg.solve(np.column_stack((a[1], -b[1])), b[0]-a[0])
            vertices.append((a[0]+factors[0]*a[1]).tolist())
        task["target_vertices_world"] = vertices
        task["target_centre_xy"] = np.mean(vertices,axis=0).tolist()
        task["target_outline_centres"] = [point.tolist() for point,_ in lines]
    if tid == 'L3-04':
        from scipy.spatial import ConvexHull
        from robot_tasks.robots.transforms import pose_transform
        pan=task['tracked']['dustpan'];inverse=np.linalg.inv(pose_transform(pan['pose']))
        cache=UsdGeom.XformCache();floor=[]
        for shape in Usd.PrimRange(stage.GetPrimAtPath(pan['path'])):
            if not shape.HasAPI(UsdPhysics.CollisionAPI) or not shape.IsA(UsdGeom.Mesh):
                continue
            if not shape.GetName().startswith(('ContinuousFloorHandle','TaperedMouth')):
                continue
            vertices=np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get())
            relative=inverse@np.asarray(cache.GetLocalToWorldTransform(shape)).T
            vertices=vertices@relative[:3,:3].T+relative[:3,3]
            if vertices[:,2].max() <= .003:
                floor.extend(vertices)
        floor=np.asarray(floor)
        if len(floor)<3:
            raise ValueError('Delivered dustpan has no low basket floor')
        xy=np.unique(floor[:,:2],axis=0)
        task['pan_floor_vertices']=xy[ConvexHull(xy).vertices].tolist()
        task['pan_floor_height_range']=[float(floor[:,2].min()),float(floor[:,2].max())]
    return task
