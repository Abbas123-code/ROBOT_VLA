"""Text-command and motion-boundary tests. Never import or run Isaac/ROS."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "isaac/scripts"))
import robot_tabletop_command as cli
import run_bottle_pick as launcher


def detection():
    return {
        "completed": True, "captured_at_unix": 1000.,
        "detector": launcher.TABLETOP_DETECTOR, "weights_sha256": launcher.TABLETOP_WEIGHTS_SHA256,
        "center_xy": [.722, .342], "tray_center_xy": [.45, -.25], "tray_floor_z": .008, "tray_rim_z": .06,
        "frames": [{
            "detector": "WongKinYiu/yolov9", "image_stamp": [999, i],
            "yolo_bottle": {"label": "Water Bottle", "class_id": 6, "confidence": .5},
            "yolo_tray": {"label": "Blue Tray", "class_id": 0, "confidence": .75},
            "center_xy": [.722, .342], "tray_center_xy": [.45, -.25], "tray_floor_z": .008,
        } for i in range(5)],
    }


def approval():
    current = detection()
    return {"completed": True, "fingerprint": "current", "mode": "validation",
            "detector": current["detector"], "weights_sha256": current["weights_sha256"],
            "vision_target_xy": current["center_xy"], "tray_center_xy": current["tray_center_xy"],
            "tray_floor_z": current["tray_floor_z"], "tray_rim_z": current["tray_rim_z"]}


class CommandTests(unittest.TestCase):
    def test_visible_validation_does_not_enable_execute(self):
        args = launcher.parse_args(["--validate", "--trained-tabletop", "--yolo-place", "--show-validation"])
        self.assertTrue(args.validate)
        self.assertTrue(args.show_validation)
        self.assertFalse(args.execute)
        for argv in (["--execute", "--trained-tabletop", "--yolo-place", "--show-validation"],
                     ["--validate", "--show-validation"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(argv)

    def test_allowlisted_variants(self):
        for text in (cli.CANONICAL_COMMAND, "PICK UP THE BOTTLE AND PLACE IT IN THE BLUE TRAY.",
                     " please  pick up bottle and put it into the blue track! ",
                     "pick up the bottle and place it in the blue track"):
            with self.subTest(text=text):
                self.assertEqual(cli.parse_command(text), cli.CANONICAL_COMMAND)

    def test_unsupported_and_shell_text_rejected(self):
        for text in ("", "pick up the can and place it in the blue tray", "release the bottle",
                     "do not " + cli.CANONICAL_COMMAND, cli.CANONICAL_COMMAND + " then repeat",
                     cli.CANONICAL_COMMAND + "; touch /tmp/injected", "$(echo bottle tray)",
                     cli.CANONICAL_COMMAND + " `id`", "bottle tray", "place bottle in red tray"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cli.parse_command(text)

    def test_cli_requires_one_explicit_mode_before_subprocess(self):
        for extra in ([], ["--validate", "--execute"], ["--listen", "--validate"]):
            with patch.object(cli.subprocess, "run") as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(["--command", cli.CANONICAL_COMMAND, *extra])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_invalid_text_never_launches(self):
        with patch.object(cli.subprocess, "run") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["--command", "pick the red block", "--execute"])
            run.assert_not_called()

    def test_safe_argv_single_call_and_exit_status(self):
        for mode in ("--validate", "--execute"):
            with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["--command", cli.CANONICAL_COMMAND, mode]), 7)
                run.assert_called_once_with(cli.launcher_argv(mode), cwd=cli.ROOT, check=False)
                argv = run.call_args.args[0]
                self.assertEqual(argv[:6], ["/bin/bash", "--noprofile", "--norc", "-c",
                                           'source /opt/ros/jazzy/setup.bash && exec "$@"', "robot-tabletop-command"])
                self.assertEqual(argv[-3:], ["--yolo-place", "--trained-tabletop", mode])
                self.assertNotIn(cli.CANONICAL_COMMAND, argv)

    def test_space_in_root_stays_one_argument(self):
        with patch.object(cli, "ROOT", Path("/tmp/root with spaces")):
            self.assertEqual(cli.launcher_argv("--validate")[6], "/tmp/root with spaces/env_isaacsim/bin/python")
        with self.assertRaises(ValueError):
            cli.launcher_argv("--execute; id")

    def test_launcher_mode_dependency_and_legacy(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["--trained-tabletop", "--validate"])
        self.assertFalse(launcher.parse_args(["--yolo-place", "--validate"]).trained_tabletop)
        self.assertFalse(launcher.parse_args(["--validate"]).yolo_place)
        self.assertTrue(launcher.parse_args(["--yolo-place", "--trained-tabletop", "--validate"]).trained_tabletop)


class MotionBoundaryTests(unittest.TestCase):
    def test_valid_contract_and_approval(self):
        launcher.validate_tabletop_detection(detection())
        launcher.require_execution_approval(approval(), detection(), "current", True)

    def test_required_report_fields(self):
        for key in detection():
            current = detection()
            del current[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launcher.validate_tabletop_detection(current)

    def test_bad_report_values(self):
        for key, value in (("completed", 1), ("detector", "official-yolov9"), ("weights_sha256", "old"),
                           ("captured_at_unix", float("nan")), ("center_xy", [True, 0]),
                           ("tray_center_xy", [float("inf"), 0]), ("tray_center_xy", [0, 0, 0]),
                           ("tray_floor_z", float("nan")), ("tray_rim_z", .007), ("frames", [])):
            current = detection()
            current[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                launcher.validate_tabletop_detection(current)

    def test_bad_frame_evidence(self):
        for item_key in ("yolo_bottle", "yolo_tray"):
            for key, value in (("confidence", .499), ("confidence", float("nan")), ("confidence", 1.1),
                               ("class_id", 39), ("class_id", False), ("label", "bottle")):
                current = detection()
                current["frames"][0][item_key][key] = value
                with self.subTest(item=item_key, key=key, value=value), self.assertRaises(RuntimeError):
                    launcher.validate_tabletop_detection(current)
        for key in detection()["frames"][0]:
            current = detection()
            del current["frames"][0][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                launcher.validate_tabletop_detection(current)

    def test_frame_stamp_and_geometry_validation(self):
        for key, value in (("image_stamp", [999, 1]), ("image_stamp", [999, 1_000_000_000]),
                           ("image_stamp", [999., 0]), ("center_xy", [.73, .34]),
                           ("tray_center_xy", [.46, -.25]), ("tray_floor_z", .02)):
            current = detection()
            current["frames"][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launcher.validate_tabletop_detection(current)

    def test_stale_future_incomplete_and_reused_capture(self):
        with patch.object(launcher.time, "time", return_value=1001.):
            launcher.require_fresh_detection(detection(), not_before=999.)
            for captured in (980., 1002., float("nan")):
                current = detection()
                current["captured_at_unix"] = captured
                with self.assertRaises(RuntimeError):
                    launcher.require_fresh_detection(current)
            with self.assertRaises(RuntimeError):
                launcher.require_fresh_detection(detection(), not_before=1000.5)
            current = detection()
            current["completed"] = False
            with self.assertRaises(RuntimeError):
                launcher.require_fresh_detection(current)

    def test_changed_bottle_or_tray_invalidates_approval(self):
        for key in ("vision_target_xy", "tray_center_xy", "tray_floor_z", "tray_rim_z"):
            previous = approval()
            if isinstance(previous[key], list):
                previous[key][0] += .0021
            else:
                previous[key] += .0021
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launcher.require_execution_approval(previous, detection(), "current", True)
        previous = approval()
        previous["vision_target_xy"][0] += .0019
        previous["tray_center_xy"][0] += .0019
        launcher.require_execution_approval(previous, detection(), "current", True)

    def test_old_invalid_or_incomplete_approvals_rejected(self):
        for key, value in (("completed", False), ("fingerprint", "old"), ("detector", "WongKinYiu/yolov9"),
                           ("weights_sha256", "old"), ("mode", "visible_execution"),
                           ("vision_target_xy", [float("nan"), 0]), ("tray_floor_z", float("nan"))):
            previous = approval()
            previous[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launcher.require_execution_approval(previous, detection(), "current", True)
        for key in ("tray_center_xy", "tray_floor_z", "tray_rim_z", "weights_sha256"):
            previous = approval()
            del previous[key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                launcher.require_execution_approval(previous, detection(), "current", True)

    def test_legacy_approval_needs_no_tray(self):
        previous = {"completed": True, "fingerprint": "current", "vision_target_xy": [.722, .342]}
        launcher.require_execution_approval(previous, {"center_xy": [.722, .342]}, "current")

    def test_fingerprint_tracks_new_dependencies_without_legacy_dependency(self):
        stage = SimpleNamespace(GetUsedLayers=lambda: [])
        dependencies = ("configs/yolov9_tabletop.json", "src/perception/yolov9_tabletop.py",
                        "src/perception/tabletop_rgbd.py", "scripts/wrist_yolo_ros.py",
                        "isaac/scripts/wrist_camera_ros.py", "scripts/robot_tabletop_command.py")
        changed = {}
        with patch.object(launcher, "tabletop_model_hash", return_value=launcher.TABLETOP_WEIGHTS_SHA256) as model:
            with patch.object(Path, "read_bytes", lambda path: changed.get(path, str(path).encode())):
                baseline = launcher.fingerprint(stage, True)
                legacy = launcher.fingerprint(stage)
                for relative in dependencies:
                    changed[ROOT / relative] = b"changed"
                    self.assertNotEqual(launcher.fingerprint(stage, True), baseline, relative)
                    self.assertEqual(launcher.fingerprint(stage), legacy)
                    changed.clear()
                model.return_value = "different model hash"
                self.assertNotEqual(launcher.fingerprint(stage, True), baseline)
                self.assertEqual(launcher.fingerprint(stage), legacy)

    def test_checkpoint_mismatch_rejected_without_deserialization(self):
        with patch.object(Path, "read_bytes", return_value=b"unapproved model"):
            with self.assertRaises(RuntimeError):
                launcher.tabletop_model_hash()


if __name__ == "__main__":
    unittest.main()