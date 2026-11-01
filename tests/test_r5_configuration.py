"""Offline asset/configuration checks; synthetic calibration is never deployed."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from robot_tasks.cli import main
from robot_tasks.config import load_config
from robot_tasks.robots.audit import audit_robot, require_robot_ready
from robot_tasks.robots.profile import OperatingLimit, load_robot_profile
from robot_tasks.robots.urdf import read_urdf


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/robots/r5.toml"
PICK_PLACE = ROOT / "configs/checks/gpt_single_arm_pick_place.toml"
HANDOVER = ROOT / "configs/tasks/l2_01_marker_uncap.toml"


class R5ConfigurationTests(unittest.TestCase):
    def test_lab_tasks_allow_native_zero_gap_for_thin_props(self):
        for path in (ROOT / "configs/tasks").glob("*.toml"):
            if path.name.endswith(".local.toml"):
                continue  # Private experiments are not part of the maintained task pool.
            config = load_config(path)
            if not config.raw["task"].get("simulation_enabled"):
                continue
            with self.subTest(task=config.task_id):
                lower = 0.0
                self.assertEqual(config.robot.gripper.width_limits_m, (lower, 0.070))
                finger_limits = {v.joint: v.lower for v in config.robot.operating_limits}
                expected_joint_lower = 0.0 if lower == 0.0 else 0.005
                for joint in config.robot.gripper.joint_names:
                    self.assertEqual(finger_limits[joint], expected_joint_lower)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.profile = load_robot_profile(PROFILE)

    def write_profile(self, before, after):
        content = PROFILE.read_text(encoding="utf-8")
        self.assertIn(before, content)
        path = self.directory / "profile.toml"
        path.write_text(content.replace(before, after), encoding="utf-8")
        return path

    def copy_assets(self):
        root = self.directory / "assets"
        shutil.copytree(ROOT / "assets/r5", root)
        return replace(
            self.profile, asset_path=root / "R5a/urdf/R5a.urdf",
            asset_lock=root / "source.lock.json", package_roots=(("R5a", root / "R5a"),),
        )

    def synthetic_ready_profile(self):
        """Only for checking validation behavior, not a suggested physical setup."""
        limits = tuple(OperatingLimit(name, -1.0, 1.0, 1.0, 0.1)
                       for name in self.profile.joint_names)
        limits += tuple(OperatingLimit(name, 0.0, 0.02, 1.0, 0.01)
                        for name in self.profile.gripper.joint_names)
        mounts = tuple(replace(
            arm, base_pose_wxyz=(0.0, float(index), 0.0, 1.0, 0.0, 0.0, 0.0),
            home_positions_rad=(0.0,) * 6,
        ) for index, arm in enumerate(self.profile.mounts))
        return replace(
            self.profile, hardware_match="TEST FIXTURE ONLY", limits_source="TEST FIXTURE ONLY",
            tcp_source="TEST FIXTURE ONLY", tcp_pose_wxyz=(0, 0, 0, 1, 0, 0, 0),
            mounts=mounts, operating_limits=limits,
            gripper=replace(self.profile.gripper, width_limits_m=(0.0, 0.02),
                            joint_scale=(0.5, 0.5), joint_offset_m=(0.0, 0.0),
                            calibration_source="TEST FIXTURE ONLY"),
        )

    def assert_invalid(self, profile, message):
        audit = audit_robot(profile)
        self.assertFalse(audit.valid)
        self.assertFalse(audit.ready)
        self.assertTrue(any(message in issue.message for issue in audit.issues), audit.issues)

    def test_pinned_vendor_asset_is_complete_but_not_execution_ready(self):
        audit = audit_robot(self.profile)
        self.assertTrue(audit.valid, audit.issues)
        self.assertTrue(audit.integrity_valid)
        self.assertFalse(audit.ready)
        self.assertFalse(audit.as_dict()["simulation_validated"])
        description = audit.description
        self.assertIsNotNone(description)
        self.assertEqual(description.name, "R5a")
        self.assertEqual(len(description.links), 9)
        self.assertEqual(len(description.meshes), 9)
        self.assertEqual(description.chain("base_link", "link6"), self.profile.joint_names)
        self.assertEqual([joint.kind for joint in description.joints],
                         ["revolute"] * 6 + ["prismatic"] * 2)
        self.assertIsNone(self.profile.tcp_pose_wxyz)
        self.assertIsNone(self.profile.gripper.width_limits_m)
        self.assertEqual(self.profile.operating_limits, ())
        for arm in self.profile.mounts:
            self.assertIsNone(arm.base_pose_wxyz)
            self.assertIsNone(arm.home_positions_rad)
        with self.assertRaisesRegex(ValueError, "not ready for execution"):
            require_robot_ready(self.profile)

    def test_tasks_select_active_arms_without_replacing_the_r5_platform(self):
        for config_path, active in ((PICK_PLACE, ("left",)), (HANDOVER, ("left", "right"))):
            with self.subTest(config=config_path.name):
                config = load_config(config_path)
                self.assertEqual(config.robot_model, "r5")
                self.assertEqual(config.robot.available_arms, ("left", "right"))
                self.assertEqual(config.arms, active)
                self.assertEqual(config.robot.asset_path, self.profile.asset_path)
                self.assertEqual(config.cameras, ("overview", "left_wrist", "right_wrist"))

    def test_asset_resolution_is_independent_of_working_directory(self):
        with chdir(self.directory):
            profile = load_robot_profile(PROFILE)
            self.assertTrue(audit_robot(profile).valid)
            self.assertEqual(profile.asset_path, self.profile.asset_path)

    def test_other_robot_model_and_legacy_schema_are_rejected(self):
        path = self.write_profile('model = "r5"', 'model = "franka"')
        with self.assertRaisesRegex(ValueError, "requires ARX R5"):
            load_robot_profile(path)
        path = self.directory / "legacy.toml"
        path.write_text(PICK_PLACE.read_text(encoding="utf-8").replace(
            "schema_version = 2", "schema_version = 1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema_version = 2"):
            load_config(path)

    def test_unknown_active_arm_and_legacy_robot_fields_are_rejected(self):
        content = PICK_PLACE.read_text(encoding="utf-8").replace(
            '"../robots/r5_sim.toml"', json.dumps(PROFILE.as_posix()))
        content = content.replace('"../workflows/gpt_default.toml"',
                                  json.dumps((ROOT / "configs/workflows/gpt_default.toml").as_posix()))
        for changed, message in (
            (content.replace('active_arms = ["left"]', 'active_arms = ["arm"]'), "subset"),
            (content.replace('[robot]', '[robot]\nmodel = "r5"'), "only profile and active_arms"),
        ):
            with self.subTest(message=message):
                path = self.directory / "task.toml"
                path.write_text(changed, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_config(path)

    def test_ambiguous_or_malformed_profile_fields_are_rejected(self):
        cases = (
            ('prim_path = "/World/R5/right"', 'prim_path = "/World/R5/left"', "distinct"),
            ('prim_path = "/World/R5/right"', 'prim_path = "/World/R5/left/child"', "nested"),
            ('[arms.left]', '[arms.left]\nbase_pose_wxyz = [0, 0, 0, 2, 0, 0, 0]', "normalized"),
            ('[gripper]', '[gripper]\nwidth_limits_m = [0.0, 0.02]', "provided together"),
            ('[arms.left]', '[arms.left]\nhome_positions_rad = [0.0]', "6 finite numbers"),
            ('[kinematics]', '[kinematics]\nend_link = "link6"', "unknown fields"),
            ('urdf/R5a.urdf', 'urdf/R5a.STEP', "CAD/STEP"),
        )
        for before, after, message in cases:
            with self.subTest(message=message):
                path = self.write_profile(before, after)
                with self.assertRaisesRegex(ValueError, message):
                    load_robot_profile(path)

    def test_incorrect_joint_order_tip_or_finger_binding_is_rejected(self):
        for profile, message in (
            (replace(self.profile, joint_names=self.profile.joint_names[::-1]), "ordered URDF chain"),
            (replace(self.profile, ee_link="missing"), "Unknown link"),
            (replace(self.profile, gripper=replace(self.profile.gripper,
                                                  joint_names=("joint7", "missing"))), "does not exist"),
        ):
            with self.subTest(message=message):
                self.assert_invalid(profile, message)

    def test_missing_mesh_and_corrupted_source_do_not_pass_preflight(self):
        profile = self.copy_assets()
        mesh = dict(profile.package_roots)["R5a"] / "meshes/link8.STL"
        original = mesh.read_bytes()
        mesh.write_bytes(original + b"unexpected modification")
        self.assert_invalid(profile, "modified or corrupt")
        mesh.unlink()
        self.assert_invalid(profile, "Missing or empty mesh")

    def test_missing_urdf_and_unmapped_package_are_reported(self):
        self.assert_invalid(replace(self.profile, asset_path=self.directory / "missing.urdf"),
                            "not covered by the source lock")
        self.assert_invalid(replace(self.profile, package_roots=()), "Unmapped mesh package")

    def test_malformed_lock_is_a_reported_error(self):
        path = self.directory / "source.lock.json"
        for content in ("[]", "null", '{"schema_version": 1, "files": {"wrong": 1}}'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                self.assert_invalid(replace(self.profile, asset_lock=path), "Invalid or empty")

    def test_complete_synthetic_configuration_can_pass_offline_preflight(self):
        audit = require_robot_ready(self.synthetic_ready_profile())
        self.assertTrue(audit.ready)
        self.assertFalse(audit.as_dict()["simulation_validated"])

    def test_home_pose_operating_limits_and_gripper_mapping_must_agree(self):
        profile = self.synthetic_ready_profile()
        mounts = (replace(profile.mounts[0], home_positions_rad=(2.0,) * 6), profile.mounts[1])
        self.assert_invalid(replace(profile, mounts=mounts), "outside operating limits")
        limits = (replace(profile.operating_limits[0], velocity=2000),) + profile.operating_limits[1:]
        self.assert_invalid(replace(profile, operating_limits=limits), "exceed the source description")
        gripper = replace(profile.gripper, joint_scale=(2.0, 2.0))
        self.assert_invalid(replace(profile, gripper=gripper), "width mapping exceeds joint limits")

    def test_cli_distinguishes_asset_validation_from_execution_readiness(self):
        for strict, expected in ((False, 0), (True, 2)):
            with self.subTest(strict=strict), redirect_stdout(io.StringIO()) as output:
                args = ["validate-robot", "--profile", str(PROFILE)]
                if strict:
                    args.append("--require-ready")
                self.assertEqual(main(args), expected)
                report = json.loads(output.getvalue())
                self.assertTrue(report["asset_and_bindings_valid"])
                self.assertFalse(report["execution_configuration_ready"])

    def test_run_rejects_unimplemented_task_before_creating_simulator_or_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "real_only.toml"
            path.write_text('schema_version = 2\n[task]\nid = "L2-07"\n'
                            'name = "offline real-only fixture"\nsimulation_enabled = false\n', encoding="utf-8")
            with patch("robot_tasks.execution.IsaacSimBackend") as backend, \
                    patch("robot_tasks.execution.GPTPolicy") as policy, redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(["run", "--config", str(path), "--policy", "gpt"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("real-only", error.getvalue())
                backend.assert_not_called()
                policy.assert_not_called()

    def test_urdf_structural_errors_are_detected_without_simulation(self):
        mutations = ("nonfinite_mass", "cyclic_link", "cyclic_mimic", "escaping_mesh")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                tree = ET.parse(self.profile.asset_path)
                root = tree.getroot()
                if mutation == "nonfinite_mass":
                    root.find("link/inertial/mass").set("value", "nan")
                    message = "finite number"
                elif mutation == "cyclic_link":
                    root.find("joint/parent").set("link", "link6")
                    message = "cycle"
                elif mutation == "cyclic_mimic":
                    joints = root.findall("joint")
                    ET.SubElement(joints[-1], "mimic", joint="joint7")
                    ET.SubElement(joints[-2], "mimic", joint="joint8")
                    message = "mimic graph contains a cycle"
                else:
                    root.find("link/visual/geometry/mesh").set("filename", "package://R5a/../outside.STL")
                    message = "escapes its package"
                path = self.directory / "broken.urdf"
                tree.write(path)
                with self.assertRaisesRegex(ValueError, message):
                    read_urdf(path, dict(self.profile.package_roots))

    def test_noncyclic_mimic_keeps_the_robot_identity(self):
        tree = ET.parse(self.profile.asset_path)
        ET.SubElement(tree.getroot().findall("joint")[-1], "mimic", joint="joint7")
        path = self.directory / "mimic.urdf"
        tree.write(path)
        description = read_urdf(path, dict(self.profile.package_roots))
        self.assertEqual(description.name, "R5a")
        self.assertEqual(description.joint("joint8").mimic, "joint7")


if __name__ == "__main__":
    unittest.main()
