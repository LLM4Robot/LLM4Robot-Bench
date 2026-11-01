"""Sequential identities must not dispatch to the task with the old same ID."""
import hashlib
import json
from pathlib import Path
import re
import unittest

from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.task_numbering import (
    RECLASSIFIED_SCHEME, from_legacy, numbering, previous_task_id, to_legacy,
)
from robot_tasks.video_delivery import scenes, setup_compatible

ROOT = Path(__file__).resolve().parents[1]


class TaskNumberingTests(unittest.TestCase):
    def test_current_catalogue_and_explicit_historical_mapping(self):
        expected = ({f"L1-{i:02}" for i in range(1, 6)} |
                    {f"L2-{i:02}" for i in range(1, 8)} |
                    {f"L3-{i:02}" for i in range(1, 5)})
        self.assertEqual({i for i in TASKS if i.startswith("L") and "_" not in i}, expected)
        self.assertEqual(len(numbering()["tasks"]), 16)
        for row in numbering()["tasks"]:
            for suffix in ("", "_v3", "_variant01"):
                self.assertEqual(from_legacy(row["legacy_id"] + suffix), row["id"] + suffix)
                self.assertEqual(to_legacy(row["id"] + suffix), row["legacy_id"] + suffix)
                self.assertEqual(resolve_task(row["id"] + suffix)[1].task_id, row["id"])
            self.assertIn("/" + row["legacy_id"] + "/", scenes()[row["id"]]["runtime_path"])
        self.assertEqual(from_legacy("L3-03"), "L3-01")
        self.assertEqual(from_legacy("L2-07"), "L2-05")
        self.assertEqual(from_legacy("L3-11"), "L2-07")
        for suffix in ("", "_v3", "_variant01"):
            self.assertEqual(previous_task_id("L2-07" + suffix), "L3-03" + suffix)
            self.assertEqual(previous_task_id("L3-03" + suffix), "L3-04" + suffix)
            self.assertEqual(previous_task_id("L3-04" + suffix), "L3-05" + suffix)
            self.assertEqual(previous_task_id("L2-07" + suffix, RECLASSIFIED_SCHEME), "L2-07" + suffix)
            self.assertEqual(previous_task_id("L3-03" + suffix, RECLASSIFIED_SCHEME), "L3-04" + suffix)
            self.assertEqual(previous_task_id("L3-04" + suffix, RECLASSIFIED_SCHEME), "L3-05" + suffix)
            with self.assertRaises((KeyError, ValueError)):
                resolve_task("L3-05" + suffix)
            self.assertIn("stacked books", resolve_task("L3-03" + suffix)[1].instruction)
            self.assertIn("blocks", resolve_task("L3-04" + suffix)[1].instruction)
        self.assertIn("whiteboard", resolve_task("L2-07")[1].instruction)
        self.assertIn("bottle", resolve_task("L3-01")[1].instruction)

    def test_previous_visual_setup_is_reused_for_the_same_physical_task(self):
        digest = "acda1a821f894899b862d23fcf3c09814cab31df670815c5229739446c6372fa"
        path = ROOT / "configs/task_references/video_visual_setup_history" / (digest + ".json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for row in numbering()["tasks"]:
            self.assertTrue(setup_compatible(row["id"], digest), row["id"])

    def test_website_configs_and_images_match_current_task_identity(self):
        rows = json.loads((ROOT / "docs/tasks.json").read_text(encoding="utf-8"))
        self.assertEqual({r["id"] for r in rows}, {r["id"] for r in numbering()["tasks"]})
        for row in rows:
            self.assertEqual(row["level"], row["id"][:2])
            for setting in row["settings"].values():
                spec, config = resolve_task(setting["id"])
                self.assertEqual(config.source, ROOT / setting["config"])
                self.assertEqual(setting["record_id"], config.experiment_id)
                for image in setting["images"].values():
                    self.assertTrue((ROOT / "docs" / image).is_file(), image)
        html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
        self.assertIn('id="task-list"', html)
        self.assertIn('src="tasks.js"', html)
        self.assertRegex(html, r'<script\b[^>]*\bsrc="app\.js(?:\?[^\"]*)?"[^>]*>')
        exported = (ROOT / "docs/tasks.js").read_text(encoding="utf-8")
        prefix = "window.BENCHMARK_TASKS = "
        self.assertTrue(exported.startswith(prefix))
        self.assertEqual(json.loads(exported[len(prefix):].strip().removesuffix(";")), rows)
        films = re.findall(r"\{id:'([^']+)'", (ROOT / "docs/films.js").read_text())
        self.assertEqual(set(films), {r["id"] for r in rows})
        for task_id in films:
            for domain in ("sim", "real"):
                for extension in ("mp4", "webp"):
                    path = ROOT / "docs/media/loops" / f"{task_id}-{domain}.{extension}"
                    self.assertTrue(path.is_file(), path)

    def test_pre_paper_alignment_setup_preserves_all_scene_identities(self):
        digest = "d186c5fc51f4553035f5dbac9b8dab84ec5176b78d02abf59a2ac2f053189628"
        path = ROOT / "configs/task_references/video_visual_setup_history" / (digest + ".json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for row in numbering()["tasks"]:
            for suffix in ("", "_v3"):
                _, cfg = resolve_task(row["id"] + suffix)
                self.assertTrue(setup_compatible(cfg.scene_id, digest), row["id"] + suffix)

    def test_pre_reclassification_visual_setup_is_still_compatible(self):
        digest = "e9fcc542e1686b725f1e74f7e572e314982f0aed83dd0bb5d876e57dda61f03c"
        path = ROOT / "configs/task_references/video_visual_setup_history" / (digest + ".json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for row in numbering()["tasks"]:
            for suffix in ("", "_v3"):
                _, cfg = resolve_task(row["id"] + suffix)
                self.assertTrue(setup_compatible(cfg.scene_id, digest), row["id"] + suffix)
