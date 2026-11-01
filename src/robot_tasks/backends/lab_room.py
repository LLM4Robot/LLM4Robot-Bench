"""A self-contained indoor workshop around the unchanged task work surface."""

import hashlib
import numpy as np

from .appearance import ASSETS, material
from .scene_geometry import BENCH_POSITION, BENCH_SIZE, BENCH_TOP

FLOOR_Z = -.78  # Keep benchmark table top at world Z=0.


def box(stage, path, centre, size, mat, *, collision=False, texture_scale=1.):
    """24-vertex box with explicit metre-scaled UVs, no external USD dependency."""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    vertices, uvs = [], []
    # Outward-facing quads, with their own seam vertices.
    faces = (((0, 0, 1), (1, 0, 0), (0, 1, 0)),
             ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
             ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
             ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    half = np.asarray(size) / 2
    for normal, u, v in faces:
        normal, u, v = np.array(normal), np.array(u), np.array(v)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(np.asarray(centre) + (normal + a * u + b * v) * half)
            uvs.append(((a + 1) * abs(u @ half) / texture_scale,
                        (b + 1) * abs(v @ half) / texture_scale))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(np.asarray(vertices, dtype=np.float32))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(list(range(24)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([np.asarray(centre) - half, np.asarray(centre) + half])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uvs)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def build_lab_room(stage):
    from pxr import Gf, UsdGeom, UsdLux

    root = "/World/Room"
    UsdGeom.Xform.Define(stage, root)
    looks = root + "/Looks/"
    wall = material(stage, looks + "Wall", (.72, .73, .70), roughness=.85)
    accent = material(stage, looks + "Accent", (.15, .23, .27), roughness=.75)
    steel = material(stage, looks + "Steel", (.065, .078, .085), roughness=.35, metallic=.7)
    aluminium = material(stage, looks + "Aluminium", (.42, .46, .49), roughness=.28, metallic=.85)
    wood = material(stage, looks + "Wood", (.5, .35, .2), textures="kitchen_wood")
    floor = material(stage, looks + "Concrete", (.4, .4, .4), textures="concrete_floor_02")
    window = material(stage, looks + "Window", (.25, .4, .55), roughness=.18,
                      emission=(.3, .45, .62))
    lamp = material(stage, looks + "Lamp", (.9, .9, .88), emission=(3., 2.9, 2.6))

    def add(name, centre, size, mat, **options):
        return box(stage, root + "/" + name, centre, size, mat, **options)

    add("Floor", (.25, 0, FLOOR_Z - .05), (5.2, 5.4, .1), floor, collision=True, texture_scale=2.)
    # The whole room is outside the robot workspace. All sensors stay inside.
    add("BackWall", (.25, 2.7, .72), (5.3, .12, 3.), wall, collision=True)
    add("FrontWall", (.25, -2.7, .72), (5.3, .12, 3.), wall, collision=True)
    add("LeftWall", (-2.35, 0, .72), (.12, 5.4, 3.), wall, collision=True)
    add("RightWall", (2.85, 0, .72), (.12, 5.4, 3.), wall, collision=True)
    add("Ceiling", (.25, 0, 2.25), (5.3, 5.5, .1), wall)
    for y in (-2.63, 2.63):
        label = "Front" if y < 0 else "Back"
        add(label + "Skirting", (.25, y, -.69), (5.15, .035, .18), accent)
    # Daylight window panes, aluminium frame and sill on the rear wall.
    add("WindowPane", (.25, 2.628, 1.0), (2.2, .018, 1.1), window)
    for i, x in enumerate((-.89, .25, 1.39)):
        add(f"WindowVertical{i}", (x, 2.6, 1.0), (.045, .075, 1.19), aluminium)
    for i, z in enumerate((.425, 1.575)):
        add(f"WindowHorizontal{i}", (.25, 2.6, z), (2.32, .075, .045), aluminium)
    add("WindowSill", (.25, 2.56, .4), (2.4, .24, .06), wall)
    # A closed door and handle; no moving joint or task interaction.
    add("Door", (-2.279, -1.45, .25), (.024, .95, 2.05), wood, texture_scale=1.5)
    add("DoorHandle", (-2.23, -1.1, .18), (.09, .14, .024), aluminium)
    for i, y in enumerate((-1.95, -.95)):
        add(f"DoorFrame{i}", (-2.25, y, .27), (.08, .05, 2.15), accent)
    add("DoorLintel", (-2.25, -1.45, 1.36), (.08, 1.05, .05), accent)

    # Replace only the bench's render surface. Keep /World/Bench, its exact
    # dimensions, friction and collider for all existing tasks and bench guards.
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Bench")).CreateVisibilityAttr("invisible")
    add("BenchSurface", BENCH_POSITION, BENCH_SIZE, wood, texture_scale=1.5)
    for i, x in enumerate((-.41, .91)):
        for j, y in enumerate((-.735, .735)):
            add(f"TableLeg_{i}_{j}", (x, y, -.415), (.065, .065, .71), steel)
            add(f"TableFoot_{i}_{j}", (x, y, -.765), (.10, .10, .025), steel)
        add(f"TableSideRail_{i}", (x, 0, -.18), (.055, 1.47, .065), steel)
    for j, y in enumerate((-.735, .735)):
        add(f"TableLongRail_{j}", (.25, y, -.12), (1.38, .055, .10), steel)

    # Low background storage gives scale without cluttering the task surface.
    add("Cabinet", (-1.85, .8, -.32), (.65, 1.45, .90), accent)
    add("CabinetTop", (-1.85, .8, .15), (.69, 1.49, .04), wood)
    for i, y in enumerate((.33, .80, 1.27)):
        add(f"CabinetDoor{i}", (-1.518, y, -.3), (.02, .44, .82), wall)
        add(f"CabinetHandle{i}", (-1.485, y + .13, -.15), (.04, .018, .15), aluminium)

    UsdLux.DomeLight.Get(stage, "/World/Light").CreateIntensityAttr(250)
    for i, y in enumerate((-.8, 1.0)):
        add(f"LightHousing{i}", (.2, y, 2.17), (1.6, .46, .055), aluminium)
        add(f"LightPanel{i}", (.2, y, 2.137), (1.5, .38, .006), lamp)
        light = UsdLux.RectLight.Define(stage, root + f"/CeilingLight{i}")
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(.2, y, 2.12))
        light.CreateWidthAttr(1.5)
        light.CreateHeightAttr(.38)
        light.CreateIntensityAttr(15000)
        light.CreateColorAttr(Gf.Vec3f(1., .95, .87))
    daylight = UsdLux.RectLight.Define(stage, root + "/WindowLight")
    transform = UsdGeom.Xformable(daylight)
    transform.AddTranslateOp().Set(Gf.Vec3d(.25, 2.56, 1.0))
    transform.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))
    daylight.CreateWidthAttr(2.1)
    daylight.CreateHeightAttr(1.0)
    daylight.CreateIntensityAttr(7000)
    daylight.CreateColorAttr(Gf.Vec3f(.83, .91, 1.))
    return {"profile": "lab_room_v1", "floor_z_m": FLOOR_Z, "bench_top_z_m": BENCH_TOP,
            "room_size_m": [5.2, 5.4, 3.], "task_coordinates": "unchanged",
            "collision_scope": "existing task bench plus distant room floor/walls; furniture is visual",
            "asset_lock_sha256": hashlib.sha256((ASSETS / "source.lock.json").read_bytes()).hexdigest(),
            "lighting": {"ceiling_rect_intensity": 15000, "window_rect_intensity": 7000,
                         "dome_intensity": 250, "randomized": False},
            "observation_variant": "new background, lighting and surface appearance; do not pool with plain runs"}
