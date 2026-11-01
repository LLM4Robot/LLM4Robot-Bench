"""Private release integrity and partial-delivery regressions; no network required."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.fetch_tuojing_release import download_archive, release_packages
from tools.tuojing_assets import select_packages, sha256, verify


class ReleaseTests(unittest.TestCase):
    def test_pinned_release_matches_current_repository_lock(self):
        root = Path(__file__).resolve().parents[1] / "assets/tuojing"
        manifest = json.loads((root / "release.json").read_text(encoding="utf-8"))
        packages = release_packages(root, manifest)
        self.assertEqual(set(packages), {p["id"] for p in manifest["packages"]})

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = b"archive fixture"
        digest = hashlib.sha256(self.payload).hexdigest()
        self.package = {"id": "present", "archive": "present.zip", "archive_sha256": digest,
                        "directory": "present", "files": {}}
        self.lock = {"delivery": "fixture", "packages": [self.package,
                     {"id": "missing", "directory": "missing", "files": {"absent": {"bytes": 1, "sha256": "none"}}}]}
        (self.root / "source.lock.json").write_text(json.dumps(self.lock))
        (self.root / "catalog.json").write_text('{"entries": []}')
        self.entry = {"id": "present", "archive": "present.zip", "bytes": len(self.payload), "sha256": digest}
        self.manifest = {"schema_version": 1, "repository": "TuojingAI/GPT_Benchmark",
                         "tag": "assets-test-v1", "source_lock_sha256": sha256(self.root / "source.lock.json"),
                         "packages": [self.entry], "omitted_packages": ["missing"]}

    def test_partial_report_does_not_claim_missing_packages_verified(self):
        result = verify(self.root, package_ids=["present"])
        self.assertEqual(result["omitted_packages"], ["missing"])
        self.assertEqual(result["verified_files"], {"present": 0})
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_selection_and_release_coverage_are_validated(self):
        self.assertEqual(list(release_packages(self.root, self.manifest)), ["present"])
        for ids in ([], ["unknown"], ["present", "present"]):
            with self.assertRaises(ValueError):
                select_packages(self.lock, ids)
        self.manifest["omitted_packages"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            release_packages(self.root, self.manifest)

    def test_changed_lock_or_archive_hash_is_rejected(self):
        self.entry["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs"):
            release_packages(self.root, self.manifest)
        (self.root / "source.lock.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source lock"):
            release_packages(self.root, self.manifest)

    @patch("tools.fetch_tuojing_release.subprocess.run")
    def test_cache_is_checked_without_network_and_modified_cache_preserved(self, run):
        target = self.root / "present.zip"
        target.write_bytes(self.payload)
        download_archive(self.root, self.entry, self.manifest, "gh")
        target.write_bytes(b"local change")
        with self.assertRaises(ValueError):
            download_archive(self.root, self.entry, self.manifest, "gh")
        self.assertEqual(target.read_bytes(), b"local change")
        run.assert_not_called()

    @patch("tools.fetch_tuojing_release.subprocess.run")
    def test_bad_remote_bytes_never_enter_cache(self, run):
        def fake_download(args, **kwargs):
            (Path(args[-1]) / "present.zip").write_bytes(b"bad remote data")
        run.side_effect = fake_download
        with self.assertRaises(ValueError):
            download_archive(self.root, self.entry, self.manifest, "gh")
        self.assertFalse((self.root / "present.zip").exists())
        self.assertFalse(list(self.root.glob(".download-*")))

    @patch("tools.fetch_tuojing_release.subprocess.run")
    def test_download_uses_pinned_repo_and_tag(self, run):
        def fake_download(args, **kwargs):
            self.assertEqual(args[3:6], ["assets-test-v1", "--repo", "TuojingAI/GPT_Benchmark"])
            (Path(args[-1]) / "present.zip").write_bytes(self.payload)
        run.side_effect = fake_download
        target = download_archive(self.root, self.entry, self.manifest, "gh")
        self.assertEqual(target.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
