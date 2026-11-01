"""Collision-only simulator geometry. Never inserted into a policy observation.

The planner is privileged to the table/fixture meshes and robot configuration.
A task may register physically measured held-body surface samples for clearance
checking; this does not attach the object in physics or establish task success.
"""

from functools import lru_cache

import numpy as np


def collider_meshes(backend):
    from pxr import Usd, UsdGeom, UsdPhysics

    from ..backends.scene_geometry import BENCH_POSITION, BENCH_SIZE

    result = {
        "cuboid": {
            "bench": {"pose": [*BENCH_POSITION, 1, 0, 0, 0], "dims": list(BENCH_SIZE)}
        } if getattr(backend, "planning_legacy_bench", True) else {},
        "mesh": {},
    }
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    task_fixture_paths = set(getattr(backend, "planning_fixture_paths", ()))
    fixture_paths = set(task_fixture_paths)
    fixture_paths.update(getattr(backend, "planning_environment_paths", ()))
    for prim in backend.world.stage.Traverse():
        path = str(prim.GetPath())
        if path not in fixture_paths or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        transform = np.array(cache.GetLocalToWorldTransform(prim)).T
        if prim.IsA(UsdGeom.Mesh):
            shape = UsdGeom.Mesh(prim)
            vertices = np.array(shape.GetPointsAttr().Get(), dtype=float)
            indices = list(shape.GetFaceVertexIndicesAttr().Get())
            faces, cursor = [], 0
            for count in shape.GetFaceVertexCountsAttr().Get():
                face = indices[cursor : cursor + count]
                cursor += count
                faces.extend(
                    [face[0], face[i], face[i + 1]] for i in range(1, count - 1)
                )
        elif prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            vertices = (
                np.array(
                    [
                        (x, y, z)
                        for x in (-0.5, 0.5)
                        for y in (-0.5, 0.5)
                        for z in (-0.5, 0.5)
                    ]
                )
                * size
            )
            faces = [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        elif prim.IsA(UsdGeom.Capsule):
            shape = UsdGeom.Capsule(prim)
            vertices, faces = capsule_mesh(shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get())
            axis = shape.GetAxisAttr().Get()
            if axis == "X":
                vertices = vertices[:, [2, 1, 0]]
                faces = np.asarray(faces)[:, ::-1].tolist()
            elif axis == "Y":
                vertices = vertices[:, [0, 2, 1]]
                faces = np.asarray(faces)[:, ::-1].tolist()
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            radius, height = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get()
            angles = np.arange(64) * 2 * np.pi / 64
            vertices = np.array(
                [
                    (radius * np.cos(a), radius * np.sin(a), z)
                    for z in (-height / 2, height / 2)
                    for a in angles
                ]
            )
            faces = []
            for i in range(64):
                j = (i + 1) % 64
                faces.extend([[i, j, 64 + j], [i, 64 + j, 64 + i]])
            faces += [[0, i + 1, i] for i in range(1, 63)] + [
                [64, 64 + i, 65 + i] for i in range(1, 63)
            ]
            axis = shape.GetAxisAttr().Get()
            if axis == "X":
                vertices = vertices[:, [2, 1, 0]]
            if axis == "Y":
                vertices = vertices[:, [0, 2, 1]]
        else:
            raise ValueError(f"Unsupported planning fixture collider: {path}")
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        result["mesh"][path.replace("/", "_")] = {
            "pose": [0, 0, 0, 1, 0, 0, 0],
            "vertices": vertices.tolist(),
            "faces": faces,
        }
    if len(task_fixture_paths) > 128:
        # Concatenate surfaces, never take their convex hull: hundreds of native
        # socket wall pieces must preserve apertures and fit the planner cache.
        names = [p.replace("/", "_") for p in sorted(task_fixture_paths)
                 if p.replace("/", "_") in result["mesh"]]
        vertices, faces = [], []
        for name in names:
            entry = result["mesh"].pop(name)
            faces.extend((np.asarray(entry["faces"]) + len(vertices)).tolist())
            vertices.extend(entry["vertices"])
        if vertices:
            result["mesh"]["task_fixture_surfaces"] = {
                "pose": [0, 0, 0, 1, 0, 0, 0], "vertices": vertices, "faces": faces}
    return result


def capsule_mesh(radius, height, segments=32, rings=16):
    """Closed Z-axis capsule surface; height is the straight portion, as in USD."""
    angles = np.arange(segments) * 2 * np.pi / segments
    polar = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)[1:-1]
    rows = []
    for p in polar:
        z = radius * np.sin(p) + (-height / 2 if p < 0 else height / 2)
        if abs(p) < 1e-8:
            rows.append([(radius * np.cos(a), radius * np.sin(a), -height / 2) for a in angles])
        rows.append([(radius * np.cos(p) * np.cos(a), radius * np.cos(p) * np.sin(a), z) for a in angles])
    vertices = np.array([v for row in rows for v in row] + [(0, 0, -height / 2 - radius), (0, 0, height / 2 + radius)])
    faces = []
    for row in range(len(rows) - 1):
        for i in range(segments):
            a, b = row * segments + i, row * segments + (i + 1) % segments
            faces.extend([[a, b, b + segments], [a, b + segments, a + segments]])
    for i in range(segments):
        j = (i + 1) % segments
        offset = (len(rows) - 1) * segments
        faces.extend([[len(vertices) - 2, j, i], [len(vertices) - 1, offset + i, offset + j]])
    return vertices, faces


def snapshot(backend):
    if not hasattr(backend, "_planner_static_scene"):
        backend._planner_static_scene = collider_meshes(backend)
    callback = getattr(backend, "planning_payloads", None)
    return {
        "obstacles": backend._planner_static_scene,
        "allow_start_collision_retreat": getattr(backend, "environment_report", {}).get("task_id") == "L3-04",
        "unexpected_contact_blocked": bool(getattr(getattr(backend, "collision_monitor", None), "blocked", None)),
        "payloads": callback() if callback is not None else {},
        "table_surface_z_m": getattr(backend, "environment_report", {}).get("table_surface_z_m", 0.0),
        "information_source": "privileged simulator collision geometry; never task targets or score",
        "limitations": "unheld movable objects excluded for contact; approximate R5 collision spheres",
    }


def box_surface(size, resolution=5):
    axes = [np.linspace(-s / 2, s / 2, resolution) for s in size]
    return np.array(
        [
            (x, y, z)
            for x in axes[0]
            for y in axes[1]
            for z in axes[2]
            if abs(abs(x) - size[0] / 2) < 1e-8
            or abs(abs(y) - size[1] / 2) < 1e-8
            or abs(abs(z) - size[2] / 2) < 1e-8
        ]
    )


@lru_cache(maxsize=1)
def _rose_surface():
    """Sample the locked physical colliders, preserving all four stem bends.

    A widened straight stem penetrates the supporting table before a pickup
    even starts. Use the same local geometry as physics/scoring, without
    changing contact tolerances or exempting the start of a trajectory.
    """
    from ..tasks.bench.native_geometry import shapes, polyhedron

    samples = []
    for index, shape in enumerate(shapes("artificial_rose")):
        vertices, _, _ = polyhedron("artificial_rose", index)
        samples.append(vertices)
        if shape["kind"] == "mesh":
            triangles = vertices[np.asarray(shape["faces"])]
            weights = np.array([(i / 4, j / 4, 1 - (i + j) / 4)
                                for i in range(5) for j in range(5 - i)])
            samples.append(np.einsum("wk,fkd->fwd", weights, triangles).reshape(-1, 3))
    points = np.unique(np.concatenate(samples), axis=0)
    points.setflags(write=False)
    return points


def article_surface(article):
    from ..robots.transforms import origin_transform

    if article.asset == "tuojing:artificial_rose":
        return _rose_surface()
    if article.asset == "tuojing:yellow_visor":
        from ..tasks.bench.native_geometry import shapes, transformed_poly

        # An enclosing box seals the head opening and makes a peg-through-band
        # trajectory impossible. Preserve every native band/brim collider.
        samples = []
        for i, shape in enumerate(shapes("yellow_visor")):
            vertices, _, _ = transformed_poly("yellow_visor", i, np.eye(4))
            triangles = vertices[np.asarray(shape["faces"])]
            weights = np.array([(i / 4, j / 4, 1 - (i + j) / 4)
                                for i in range(5) for j in range(5 - i)])
            samples.append(np.einsum("wk,fkd->fwd", weights, triangles).reshape(-1, 3))
        return np.concatenate(samples)
    if article.asset and not article.parts:
        # Conservative box for imported objects, disclosed as such in planner logs.
        if article.asset_bounds:
            low, high = np.asarray(article.asset_bounds)
            return box_surface(high - low) + (low + high) / 2
        return box_surface(article.asset_size)
    points = []
    for part in article.parts:
        if not part.collision:
            continue
        if part.shape == "box":
            local = box_surface(part.size)
        elif part.shape == "mesh":
            triangles = np.asarray(part.vertices)[np.asarray(part.faces)]
            weights = np.array([(i / 4, j / 4, 1 - (i + j) / 4)
                                for i in range(5) for j in range(5 - i)])
            local = np.einsum("wk,fkd->fwd", weights, triangles).reshape(-1, 3)
        elif part.shape == "sphere":
            azimuth = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            polar = np.linspace(0, np.pi, 17)
            local = np.array([(part.size[0] * np.sin(p) * np.cos(a),
                               part.size[0] * np.sin(p) * np.sin(a), part.size[0] * np.cos(p))
                              for p in polar for a in azimuth])
        else:
            radius, _, height = part.size
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            local = np.array(
                [
                    (r * np.cos(a), r * np.sin(a), z)
                    for z in np.linspace(-height / 2, height / 2, 7)
                    for r in (0, radius / 2, radius)
                    for a in angles
                    if r == radius or abs(abs(z) - height / 2) < 1e-8
                ]
            )
        t = origin_transform(part.position, part.rotation)
        points.extend(local @ t[:3, :3].T + t[:3, 3])
    return np.asarray(points)
