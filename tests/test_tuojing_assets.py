"""Offline asset-import regressions; no SSH, private assets, Isaac, or model calls."""

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.tuojing_assets import install_package, relative_path, sha256, verify_files, verify_usd

try:
    from pxr import Usd
except ImportError:
    Usd = None


class TuojingAssetImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "delivery.zip"
        self.content = b"texture fixture"
        self.package = {
            "archive": self.archive.name,
            "archive_root": "original_folder",
            "directory": "delivery",
            "files": {"textures/sample.bin": {"bytes": len(self.content),
                       "sha256": hashlib.sha256(self.content).hexdigest()}},
        }
        self.make_archive({"original_folder/textures/sample.bin": self.content})

    def make_archive(self, files):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        self.package["archive_sha256"] = sha256(self.archive)

    def test_folder_mapping_and_reinstall_preserve_the_delivery(self):
        install_package(self.root, self.root, self.package)
        target = self.root / "packages/delivery"
        self.assertEqual(verify_files(target, self.package), 1)
        self.archive.unlink()
        install_package(self.root, self.root, self.package)
        self.assertEqual((target / "textures/sample.bin").read_bytes(), self.content)

    def test_modified_archive_never_installs(self):
        with self.archive.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            install_package(self.root, self.root, self.package)
        self.assertFalse((self.root / "packages/delivery").exists())

    def test_explicit_rootless_delivery_retains_inventory_and_path_checks(self):
        self.package["archive_root"] = ""
        self.make_archive({"textures/sample.bin": self.content})
        install_package(self.root, self.root, self.package)
        self.assertEqual(verify_files(self.root / "packages/delivery", self.package), 1)
        self.package["directory"] = "unsafe"
        self.make_archive({"../escaped.bin": self.content})
        with self.assertRaisesRegex(ValueError, "Unsafe asset path"):
            install_package(self.root, self.root, self.package)
        self.assertFalse((self.root / "escaped.bin").exists())

    def test_path_traversal_is_rejected_even_with_a_matching_archive_hash(self):
        self.make_archive({"original_folder/../../escaped.bin": self.content})
        with self.assertRaisesRegex(ValueError, "Unsafe asset path"):
            install_package(self.root, self.root, self.package)
        self.assertFalse((self.root / "escaped.bin").exists())
        self.assertEqual(list((self.root / "packages").iterdir()), [])

    def test_alternative_wrapper_uses_same_inventory(self):
        alternative = self.root / "original.zip"
        with zipfile.ZipFile(alternative, "w") as archive:
            archive.writestr("textures/sample.bin", self.content)
        self.package["archive_alternatives"] = [{
            "archive": alternative.name, "archive_root": "",
            "archive_sha256": sha256(alternative),
        }]
        self.archive.unlink()
        install_package(self.root, self.root, self.package)
        self.assertEqual(verify_files(self.root / "packages/delivery", self.package), 1)

    def test_corrupt_primary_does_not_fall_back_to_alternative(self):
        alternative = self.root / "original.zip"
        alternative.write_bytes(self.archive.read_bytes())
        self.package["archive_alternatives"] = [{
            "archive": alternative.name, "archive_root": "original_folder",
            "archive_sha256": sha256(alternative),
        }]
        self.archive.write_bytes(b"corrupt primary")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            install_package(self.root, self.root, self.package)
        self.assertFalse((self.root / "packages/delivery").exists())

    def test_bad_extracted_content_leaves_no_partial_install(self):
        self.make_archive({"original_folder/textures/sample.bin": b"wrong"})
        with self.assertRaisesRegex(ValueError, "Missing or modified"):
            install_package(self.root, self.root, self.package)
        self.assertEqual(list((self.root / "packages").iterdir()), [])

    def test_existing_modified_delivery_is_not_overwritten(self):
        install_package(self.root, self.root, self.package)
        target = self.root / "packages/delivery/textures/sample.bin"
        target.write_bytes(b"local change")
        with self.assertRaisesRegex(ValueError, "Missing or modified"):
            install_package(self.root, self.root, self.package)
        self.assertEqual(target.read_bytes(), b"local change")

    def test_paths_are_portable_on_windows_and_linux(self):
        for value in ("/root/file", "C:/file", "../file", "a/../../file", "a\\file", "."):
            with self.subTest(path=value), self.assertRaises(ValueError):
                relative_path(value)

    @unittest.skipIf(Usd is None, "Requires standalone USD Python libraries")
    def test_scene_without_default_prim_is_reported_but_not_accepted_as_a_component(self):
        path = self.root / "scene.usda"
        path.write_text('#usda 1.0\n(metersPerUnit = 1\nupAxis = "Z")\ndef Xform "World" {}\n',
                        encoding="utf-8")
        result = verify_usd(path, self.root)
        self.assertIsNone(result["default_prim"])
        self.assertTrue(result["notes"])
        with self.assertRaisesRegex(ValueError, "default prim"):
            verify_usd(path, self.root, require_component=True)

    @unittest.skipIf(Usd is None, "Requires standalone USD Python libraries")
    def test_missing_usd_dependency_is_not_reported_as_verified(self):
        path = self.root / "scene.usda"
        path.write_text('#usda 1.0\n(subLayers = [@missing.usda@])\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unresolved USD dependencies"):
            verify_usd(path, self.root)


if __name__ == "__main__":
    unittest.main()
