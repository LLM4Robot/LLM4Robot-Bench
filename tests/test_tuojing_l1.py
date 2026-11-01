"""Task semantics after native prop migration; no private assets or model calls."""

from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from robot_tasks.planning.scene import article_surface, capsule_mesh, collider_meshes
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.scoring_l1 import cups, flower, plug
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.l1.specs import pose
from test_household_tasks import measurement

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class NativeTaskScoringTests(unittest.TestCase):
    def test_marker_photo_layout_and_native_ceramic_cavity(self):
        from robot_tasks.robots.transforms import pose_transform
        from robot_tasks.tasks.bench.native_geometry import cup_cavity, cavity_fit

        spec = build_spec("L1-01")
        mug = pose_transform(spec.design.fixture_pose)
        pen = spec.design.articles[0]
        self.assertEqual(spec.design.fixture_asset, "tuojing:black_ceramic_mug")
        # From the table front, +X is image right and -Y points toward the viewer.
        self.assertGreater(pen.initial[0], mug[0, 3])
        self.assertAlmostEqual(pen.initial[1], mug[1, 3])
        np.testing.assert_allclose(mug[:3, 0], [0, -1, 0], atol=1e-8)
        planes, floor, rim = cup_cavity('black_ceramic_mug')
        self.assertAlmostEqual(floor, .007)
        self.assertAlmostEqual(rim, .090)
        # The mug bulges below its mouth. A single cylinder/frustum would lose this.
        bulge = np.array([.045, 0, .045])
        self.assertGreater(np.min(planes[:,3] - planes[:,:3] @ bulge), 0)
        fit = lambda t: cavity_fit('whiteboard_marker', t, .001, receptacle='black_ceramic_mug')[0]
        self.assertTrue(fit(pose_transform(pose(0, 0, floor))))
        # Fits this mug, but not the old paper cup. No enlargement of tolerance.
        near_rim = pose_transform(pose(.031, 0, .075))
        self.assertTrue(fit(near_rim))
        self.assertFalse(cavity_fit('whiteboard_marker', near_rim, .001)[0])
        for xyz in ((0, 0, .004), (0, 0, .10), (.045, 0, .075), (.06, 0, .007)):
            with self.subTest(xyz=xyz):
                self.assertFalse(fit(pose_transform(pose(*xyz))))

    def test_rigid_rose_requires_release_support_and_native_stem_inside_cup(self):
        spec = build_spec("L1-05")
        article, = spec.design.articles
        self.assertEqual(article.asset, "tuojing:artificial_rose")
        self.assertEqual(spec.joints, ())
        self.assertEqual(spec.design.fixture_asset, "tuojing:ripple_paper_cup")
        e = BenchEvaluator(spec, {article.name: article.initial})
        rows = measurement(spec)
        r = rows["rose"]
        r["pose"] = list(article.goal)
        r["fixture_force_n"], r["bench_force_n"] = .07848, 0.
        self.assertTrue(flower(e, rows)[0])
        e.grasped.add("rose")
        self.assertTrue(flower(e, rows)[0])
        for delta in ([.05, 0, 0], [0, 0, spec.parameters["rim"] - article.goal[2] + .005]):
            r["pose"][:3] = (np.asarray(article.goal[:3]) + delta).tolist()
            self.assertFalse(flower(e, rows)[0])
        r["pose"] = list(article.goal)
        r["fingers"]["left"] = [1, 1]
        self.assertFalse(flower(e, rows)[0])
        r["fingers"]["left"] = [0, 0]
        r["fixture_force_n"] = 0
        self.assertFalse(flower(e, rows)[0])

    def test_rigid_rose_allows_leaning_and_uses_lift_verification(self):
        from robot_tasks.registry import resolve_task

        task, config = resolve_task("L1-05")
        self.assertEqual(task.factory.prepare_config(config).grasp_check_mode, "lift")
        spec = build_spec("L1-05")
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        rows = measurement(spec)
        r = rows["rose"]
        # The real 2.1-mm stem may rest near the inner wall; using its larger
        # planner-clearance envelope for scoring would reject this placement.
        r["pose"] = list(pose(spec.parameters["vase_xy"][0] + .024,
                              spec.parameters["vase_xy"][1], spec.parameters["floor"] + .001,
                              pitch=-np.pi / 6))
        r["fixture_force_n"], r["bench_force_n"] = .07848, 0.
        e.grasped.add("rose")
        self.assertTrue(flower(e, rows)[0])

    def test_two_native_blades_need_both_slots_but_not_full_depth(self):
        spec = build_spec("L1-02")
        article = spec.design.articles[0]
        e = BenchEvaluator(spec, {"plug": article.initial})
        rows = measurement(spec)
        r = rows["plug"]
        r["pose"] = list(article.goal)
        r["fixture_force_n"], r["bench_force_n"] = .54, 0
        self.assertTrue(plug(e, rows)[0])
        e.grasped.add("plug")
        self.assertTrue(plug(e, rows)[0])
        # The pair is symmetric at 180 degrees, but not 90 degrees.
        for yaw, expected in ((3 * np.pi / 2, True), (0, False)):
            r["pose"] = list(pose(*article.goal[:3], yaw=yaw))
            self.assertEqual(plug(e, rows)[0], expected)
        r["pose"] = list(article.goal)
        r["pose"][0] += .003
        self.assertFalse(plug(e, rows)[0])
        r["pose"] = list(article.goal)
        r["pose"][2] += .01
        self.assertTrue(plug(e, rows)[0])
        r["pose"][2] += .01
        self.assertFalse(plug(e, rows)[0])

    def test_charger_in_source_holder_is_incomplete_and_tabletop_recovery_allowed(self):
        spec = build_spec("L1-02")
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        rows = measurement(spec)
        for now in (.1, .5, 2.):
            self.assertEqual(e.update(now, rows).status, "running")
        rows["plug"]["pose"] = list(pose(.35, 1.20, .8055, pitch=np.pi / 2))
        self.assertEqual(e.update(2.2, rows).status, "running")
        rows["plug"]["corners"] = [[0, 0, .5]]
        self.assertEqual(e.update(2.5, rows).status, "failed")

    def test_charger_photo_layout_preserves_free_holder_and_downward_blades(self):
        from robot_tasks.robots.transforms import pose_transform

        spec = build_spec("L1-02")
        charger, base = spec.design.articles
        strip = pose_transform(spec.design.fixture_pose)
        self.assertEqual(base.asset, "tuojing:plug_storage_base")
        self.assertAlmostEqual(base.mass, .0208)
        self.assertEqual(spec.joints, ())
        self.assertGreater(base.initial[0], strip[0, 3])
        np.testing.assert_allclose(charger.initial[:2], base.initial[:2])
        np.testing.assert_allclose(strip[:3, 1], [-1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(pose_transform(charger.initial)[:3, 2], [0, 0, 1])
        self.assertGreater(charger.initial[2] + .018, base.initial[2] + .020)
        self.assertLess(charger.initial[2], base.initial[2] + .020)

    def test_cup_rim_balancing_and_interpenetration_do_not_count_as_nesting(self):
        spec = build_spec("L1-04")
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        e.grasped.add("upper_cup")
        rows = measurement(spec)
        top, base = rows["upper_cup"], rows["lower_cup"]
        top["pair_forces"]["lower_cup"] = .137
        base["bench_force_n"] = .274
        for height, expected in ((.025, True), (.1075, False), (.003, False)):
            top["pose"] = list(base["pose"])
            top["pose"][2] += height
            self.assertEqual(cups(e, rows)[0], expected)

    def test_native_charger_payload_does_not_fill_space_between_blades(self):
        article = build_spec("L1-02").design.articles[0]
        samples = article_surface(article)
        self.assertFalse(np.any((samples[:, 2] < .016) & (np.abs(samples[:, 0]) < .005)))

    def test_nesting_uses_cavity_fit_instead_of_an_ideal_pose_band(self):
        from robot_tasks.robots.transforms import pose_transform, transform_pose

        spec = build_spec("L1-04")
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        e.grasped.add("upper_cup")
        rows = measurement(spec)
        top, base = rows["upper_cup"], rows["lower_cup"]
        top["pair_forces"]["lower_cup"] = .137
        base["bench_force_n"] = .274
        base["pose"] = list(pose(*base["pose"][:3], pitch=.10))
        # More than 3 mm offset and 35 mm separation can still fit a tapered cavity.
        top["pose"] = list(transform_pose(pose_transform(base["pose"])
                                         @ pose_transform(pose(.004, 0, .05))))
        self.assertTrue(cups(e, rows)[0])
        top["pose"][0] += .03
        self.assertFalse(cups(e, rows)[0])

    def test_toppled_cup_on_table_is_not_a_terminal_failure(self):
        from robot_tasks.robots.transforms import pose_transform
        from robot_tasks.tasks.l1.specs import local_corners

        for task_id, name in (("L1-04", "upper_cup"),):
            spec = build_spec(task_id)
            rows = measurement(spec)
            article = next(a for a in spec.design.articles if a.name == name)
            row = rows[name]
            row["pose"] = list(pose(*article.initial[:2], spec.parameters["table_z"] + .06,
                                    pitch=np.pi / 2))
            t = pose_transform(row["pose"])
            row["corners"] = (local_corners(article) @ t[:3, :3].T + t[:3, 3]).tolist()
            e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
            for now in (.1, .5, 1.):
                self.assertEqual(e.update(now, rows).status, "running")

    def test_rose_shallow_entry_requires_real_support_not_arbitrary_depth(self):
        spec = build_spec("L1-05")
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        e.grasped.add("rose")
        rows = measurement(spec)
        row = rows["rose"]
        row["pose"] = list(pose(*spec.parameters["vase_xy"], spec.parameters["rim"] - .01))
        row["bench_force_n"] = 0
        row["fixture_force_n"] = .07848
        self.assertTrue(flower(e, rows)[0])
        row["fixture_force_n"] = 0
        self.assertFalse(flower(e, rows)[0])

    def test_capsule_surface_is_closed_and_has_correct_extent(self):
        vertices, faces = capsule_mesh(.01, .1)
        edges = Counter(tuple(sorted((face[i], face[(i + 1) % 3])))
                        for face in faces for i in range(3))
        self.assertTrue(all(count == 2 for count in edges.values()))
        np.testing.assert_allclose(vertices.min(0), [-.01, -.01, -.06], atol=1e-9)
        np.testing.assert_allclose(vertices.max(0), [.01, .01, .06], atol=1e-9)
        triangles = vertices[np.asarray(faces)]
        volume = np.sum(np.einsum("ij,ij->i", triangles[:, 0],
                                  np.cross(triangles[:, 1], triangles[:, 2]))) / 6
        self.assertGreater(volume, 0)

    @unittest.skipIf(Usd is None, "Requires optional usd-core")
    def test_many_fixed_shapes_are_concatenated_without_filling_the_gaps(self):
        stage = Usd.Stage.CreateInMemory()
        paths = []
        for i in range(129):
            shape = UsdGeom.Cube.Define(stage, f"/Fixture/Wall_{i}")
            shape.CreateSizeAttr(.01)
            shape.AddTranslateOp().Set(Gf.Vec3d(i * .02, 0, 0))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            paths.append(str(shape.GetPath()))
        result = collider_meshes(SimpleNamespace(world=SimpleNamespace(stage=stage),
                                  planning_fixture_paths=paths, planning_legacy_bench=False))
        self.assertEqual(len(result["mesh"]), 1)
        mesh = next(iter(result["mesh"].values()))
        triangles = np.asarray(mesh["vertices"])[np.asarray(mesh["faces"])]
        self.assertEqual(len(triangles), 129 * 12)
        self.assertLess(np.max(np.ptp(triangles[:, :, 0], axis=1)), .010001)


if __name__ == "__main__":
    unittest.main()
