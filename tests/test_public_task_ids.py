"""Release aliases preserve every task and reject duplicate experiments."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from robot_tasks.batch import create_batch
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.task_ids import internal_task_id, public_task_id


class PublicTaskIdTests(unittest.TestCase):
    def test_all_aliases_resolve_identical_configuration(self):
        variants = [key for key in TASKS if key.endswith("_v3")]
        self.assertEqual(len(variants), 16)
        for key in variants:
            with self.subTest(task=key):
                alias = public_task_id(key)
                self.assertEqual(internal_task_id(alias), key)
                self.assertEqual(resolve_task(alias), resolve_task(key))
                spec, config = resolve_task(alias)
                self.assertEqual(config.experiment_id, key)
                self.assertEqual(spec.factory.prepare_config(config),
                                 spec.factory.prepare_config(resolve_task(key)[1]))

    def test_alias_and_legacy_are_one_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unique"):
                create_batch(Path(tmp) / "batch",
                             SimpleNamespace(tasks=["L2-03_v3", "L2-03_variant01"]),
                             None, None)

    def test_unknown_variants_and_config_mismatches_fail(self):
        for name in ("L1-99_variant01", "L2-03_variant02", "UNKNOWN_variant01"):
            with self.subTest(task=name), self.assertRaises(ValueError):
                resolve_task(name)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_task("L2-03_variant01", TASKS["L2-03"].default_config)
