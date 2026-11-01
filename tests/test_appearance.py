"""Check asset alignment and isolate rendering changes from robot physics."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from robot_tasks.backends.appearance import ASSETS, VISUAL_OFFSETS, asset_path, glb_primitives, apply_r5_appearance
from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.config import load_config

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux
except ImportError:
    Usd = None

ROOT = Path(__file__).resolve().parents[1]


class AppearanceAssetsTests(unittest.TestCase):
    def test_pinned_assets_and_robot_alignment(self):
        lock = json.loads((ASSETS / "source.lock.json").read_text())
        for entry in lock["files"]:
            self.assertEqual(asset_path(entry["path"]).stat().st_size, entry["bytes"])
        # Binary STL has 50 bytes per triangle: normal, three vertices, padding.
        dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("padding", "<u2")])
        for glb in sorted((ASSETS / "r5").glob("*.glb")):
            with self.subTest(link=glb.stem):
                data = (ROOT / "assets/r5/R5a/meshes" / (glb.stem + ".STL")).read_bytes()
                original = np.unique(np.frombuffer(data, dtype=dtype, offset=84)["vertices"].reshape(-1, 3), axis=0)
                visual = np.concatenate([p["points"] for p in glb_primitives(glb)])
                visual += VISUAL_OFFSETS.get(glb.stem, np.zeros(3))
                np.testing.assert_allclose([visual.min(0), visual.max(0)], [original.min(0), original.max(0)], atol=1e-6)
                # Check interior vertices too, in both directions, not just bounds.
                for source, target in ((original, visual), (visual, original)):
                    sampled = source[np.linspace(0, len(source) - 1, 80).astype(int)]
                    distances = np.linalg.norm(sampled[:, None, :] - target[None, :, :], axis=2)
                    self.assertLess(distances.min(axis=1).max(), 1e-6)

    def test_profiles_are_explicit_and_legacy_default_is_preserved(self):
        config = load_config(ROOT / "configs/tasks/l1_01_cylinder_insertion.toml")
        self.assertEqual(SimulationSettings.from_config(config).environment, "tuojing_lab_v4")
        raw = dict(config.raw)
        raw["simulation"] = dict(raw["simulation"])
        raw["simulation"].pop("environment")
        raw["simulation"].pop("robot_appearance", None)
        legacy = SimulationSettings.from_config(replace(config, raw=raw))
        self.assertEqual((legacy.environment, legacy.robot_appearance), ("plain", "vendor_white"))
        for field in ("environment", "robot_appearance"):
            invalid = {**raw, "simulation": {**raw["simulation"], field: "misspelled"}}
            with self.assertRaises(ValueError):
                SimulationSettings.from_config(replace(config, raw=invalid))


@unittest.skipIf(Usd is None, "Requires USD Python libraries")
class AppearanceStageTests(unittest.TestCase):
    def test_visual_overlay_does_not_modify_existing_physics(self):
        stage = Usd.Stage.CreateInMemory()
        colliders = []
        for link in ["base_link"] + [f"link{i}" for i in range(1, 9)]:
            root = "/Robot/" + link
            prim = UsdGeom.Xform.Define(stage, root).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.5)
            UsdGeom.Xform.Define(stage, root + "/visuals")
            collider = UsdGeom.Cube.Define(stage, root + "/collisions")
            collider.CreateSizeAttr(.03)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            colliders.append(collider.GetPrim())
        before = [(p.GetPath(), str(p.GetAttributes()), p.GetAttribute("size").Get(), p.GetAppliedSchemas()) for p in colliders]
        report = apply_r5_appearance(stage, "/Robot")
        self.assertGreaterEqual(report["visual_meshes"], 9)
        self.assertEqual(before, [(p.GetPath(), str(p.GetAttributes()), p.GetAttribute("size").Get(), p.GetAppliedSchemas()) for p in colliders])
        self.assertEqual(sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()), 9)
        for prim in stage.Traverse():
            if "/Appearance/" in str(prim.GetPath()):
                self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath("/Robot/link5/visuals")).ComputeVisibility(), "invisible")

    def test_room_has_uvs_and_preserves_task_bench(self):
        from robot_tasks.backends.lab_room import build_lab_room
        from robot_tasks.backends.scene_geometry import BENCH_POSITION, BENCH_SIZE
        stage = Usd.Stage.CreateInMemory()
        bench = UsdGeom.Cube.Define(stage, "/World/Bench")
        bench.CreateSizeAttr(1.)
        UsdPhysics.CollisionAPI.Apply(bench.GetPrim())
        UsdLux.DomeLight.Define(stage, "/World/Light")
        report = build_lab_room(stage)
        self.assertEqual(report["bench_top_z_m"], 0.)
        self.assertTrue(bench.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(bench.GetSizeAttr().Get(), 1.)
        surface = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Room/BenchSurface"))
        pts = np.asarray(surface.GetPointsAttr().Get())
        np.testing.assert_allclose(pts.max(0) - pts.min(0), BENCH_SIZE, atol=1e-7)
        np.testing.assert_allclose((pts.max(0) + pts.min(0)) / 2, BENCH_POSITION, atol=1e-7)
        self.assertEqual(len(UsdGeom.PrimvarsAPI(surface).GetPrimvar("st").Get()), 24)
        self.assertFalse(surface.GetPrim().HasAPI(UsdPhysics.CollisionAPI))


if __name__ == "__main__":
    unittest.main()
