"""Stdlib-only tests; never launch Isaac or write into live scenes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bundle", ROOT / "scripts/export_portable_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class PortableBundleTests(unittest.TestCase):
    def test_rewrites_internal_reference_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene = root / "scenes/scene.usda"
            asset = root / "assets/model.usda"
            scene.parent.mkdir()
            asset.parent.mkdir()
            asset.write_text("#usda 1.0\n")
            original = f"#usda 1.0\n@{asset}@\n@https://example.com/sensor.usd@\n".encode()
            scene.write_bytes(original)
            with patch.object(bundle, "ROOT", root):
                exported, edits = bundle.portable_usda(scene, original)
            self.assertIn(b"@../assets/model.usda@", exported)
            self.assertIn(b"@https://example.com/sensor.usd@", exported)
            self.assertEqual(len(edits), 1)
            self.assertEqual(scene.read_bytes(), original)

    def test_external_local_reference_is_rejected(self):
        with patch.object(bundle, "ROOT", Path("/project")):
            with self.assertRaises(RuntimeError):
                bundle.portable_usda(Path("/project/scene.usda"), b"@/outside/model.usda@")

    def test_missing_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(bundle, "ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    bundle.portable_usda(root / "scene.usda", f"@{root}/missing.usda@".encode())


if __name__ == "__main__":
    unittest.main()