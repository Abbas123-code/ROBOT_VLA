"""Validate or execute an image-targeted bottle pick in a separate simulation.

Gets a fresh ROS image estimate from the domain-44 frozen preview before launch.
Control uses domain45 (pickup) or domain46 (YOLO placement); no command is sent
to an existing scene.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws/src/vla_sim"))
sys.path.insert(0, str(ROOT))
SCENE = ROOT / "isaac/scenes/panda_tabletop_cluttered.usda"
URDF = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
REPORTS = ROOT / "ros2_ws/log"
TABLETOP_DETECTOR = "trained-yolov9-tabletop"
TABLETOP_WEIGHTS_SHA256 = "28db4ee45eb19eb6d44d7cb9bcc8d0c13b8a4fda95f57dcd89dd508ef9910841"
TABLETOP_WEIGHTS = ROOT / "outputs/training/tabletop_yolov9s_v1/weights/best.pt"


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def require_xy(value, name):
    if not isinstance(value, list) or len(value) != 2 or not all(map(finite_number, value)):
        raise RuntimeError(f"Invalid {name}: finite two-element XY required")
    return value


def require_fresh_detection(detection, max_age=15, not_before=None):
    captured = detection.get("captured_at_unix")
    if (detection.get("completed") is not True or not finite_number(captured)
            or not 0 <= time.time() - captured <= max_age
            or (not_before is not None and captured < not_before)):
        raise RuntimeError("Fresh vision estimate required")


def validate_tabletop_detection(detection):
    """Validate the motion-boundary contract, without ROS/Isaac imports.

    Source publisher timestamp matching belongs to the capture process. Here
    require five distinct image stamps and finite, consistent world targets.
    """
    if (detection.get("completed") is not True
            or detection.get("detector") != TABLETOP_DETECTOR
            or detection.get("weights_sha256") != TABLETOP_WEIGHTS_SHA256
            or not finite_number(detection.get("captured_at_unix"))):
        raise RuntimeError("Completed pinned trained-tabletop detection required")
    for key in ("center_xy", "tray_center_xy"):
        require_xy(detection.get(key), key)
    floor, rim = detection.get("tray_floor_z"), detection.get("tray_rim_z")
    if not all(map(finite_number, (floor, rim))) or rim <= floor:
        raise RuntimeError("Finite detected tray floor and higher rim required")
    frames = detection.get("frames")
    if not isinstance(frames, list) or len(frames) != 5:
        raise RuntimeError("Five trained-tabletop observations required")
    stamps = set()
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("detector") != "WongKinYiu/yolov9":
            raise RuntimeError("Official WongKinYiu/yolov9 frame evidence required")
        for key, label, class_id in (("yolo_bottle", "Water Bottle", 6), ("yolo_tray", "Blue Tray", 0)):
            item = frame.get(key)
            if (not isinstance(item, dict) or item.get("label") != label
                    or type(item.get("class_id")) is not int or item["class_id"] != class_id
                    or not finite_number(item.get("confidence")) or not .5 <= item["confidence"] <= 1):
                raise RuntimeError(f"Invalid trained-tabletop {key} evidence")
        stamp = frame.get("image_stamp")
        if (not isinstance(stamp, list) or len(stamp) != 2
                or any(type(v) is not int for v in stamp) or stamp[0] < 0
                or not 0 <= stamp[1] < 1_000_000_000 or tuple(stamp) in stamps):
            raise RuntimeError("Five distinct ROS image stamps required")
        stamps.add(tuple(stamp))
        for key in ("center_xy", "tray_center_xy"):
            xy = require_xy(frame.get(key), key)
            if math.dist(xy, detection[key]) > .003:
                raise RuntimeError(f"Unstable or inconsistent {key}")
        if not finite_number(frame.get("tray_floor_z")) or abs(frame["tray_floor_z"] - floor) > .003:
            raise RuntimeError("Unstable or inconsistent tray floor")


def require_execution_approval(validation, detection, digest, trained_tabletop=False):
    if validation.get("completed") is not True or validation.get("fingerprint") != digest:
        raise RuntimeError("Current successful isolated validation required")
    old_bottle = require_xy(validation.get("vision_target_xy"), "validated bottle target")
    if math.dist(old_bottle, require_xy(detection.get("center_xy"), "bottle target")) > .002:
        raise RuntimeError("Bottle image target changed since isolated validation")
    if trained_tabletop:
        validate_tabletop_detection(detection)
        if (validation.get("detector") != TABLETOP_DETECTOR
                or validation.get("weights_sha256") != TABLETOP_WEIGHTS_SHA256
                or validation.get("mode") != "validation"):
            raise RuntimeError("Trained-tabletop validation required; old approvals are not reusable")
        old_xy = require_xy(validation.get("tray_center_xy"), "validated tray target")
        old_floor, old_rim = validation.get("tray_floor_z"), validation.get("tray_rim_z")
        if not all(map(finite_number, (old_floor, old_rim))):
            raise RuntimeError("Validated tray floor/rim missing or invalid")
        if (math.dist([*old_xy, old_floor], [*detection["tray_center_xy"], detection["tray_floor_z"]]) > .002
                or abs(old_rim - detection["tray_rim_z"]) > .002):
            raise RuntimeError("Tray image target changed since isolated validation")


def tabletop_model_hash():
    digest = hashlib.sha256(TABLETOP_WEIGHTS.read_bytes()).hexdigest()
    if digest != TABLETOP_WEIGHTS_SHA256:
        raise RuntimeError("Trained-tabletop checkpoint hash mismatch")
    return digest


def fingerprint(stage, trained_tabletop=False):
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("bottle_pick_controller.py"),
                 Path(__file__).with_name("yolo_bottle_place_controller.py"),
                 Path(__file__).with_name("pick_place_controller.py"), URDF,
                 ROOT / "src/perception/bottle_rgbd.py", ROOT / "scripts/detect_bottle_rgbd.py",
                 ROOT / "src/perception/yolov9_official.py", ROOT / "configs/yolov9_official.json",
                 ROOT / "src/control/bottle_grasp_plan.py",
                 ROOT / "src/control/contact_evidence.py",
                 ROOT / "ros2_ws/src/vla_sim/vla_sim/kinematics.py", ROOT / "ros2_ws/src/vla_sim/vla_sim/motion.py"):
        digest.update(path.read_bytes())
    if trained_tabletop:
        digest.update(TABLETOP_DETECTOR.encode())
        digest.update(tabletop_model_hash().encode())
        for relative in ("configs/yolov9_tabletop.json", "src/perception/yolov9_tabletop.py",
                         "src/perception/tabletop_rgbd.py", "scripts/wrist_yolo_ros.py",
                         "isaac/scripts/live_motion_camera.py",
                         "isaac/scripts/wrist_camera_ros.py", "scripts/robot_tabletop_command.py"):
            digest.update(relative.encode())
            digest.update((ROOT / relative).read_bytes())
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.identifier):
        if layer != stage.GetSessionLayer():
            digest.update(layer.ExportToString().encode())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--yolo-place", action="store_true", help="Official YOLOv9 body grasp and box placement (domain46)")
    parser.add_argument("--trained-tabletop", action="store_true",
                        help="Use the pinned trained bottle + tray RGB-D detector; requires --yolo-place")
    parser.add_argument("--show-validation", action="store_true",
                        help="Show guarded validation in a GUI with domain-46 moving wrist RGB; requires --validate --trained-tabletop")
    args = parser.parse_args(argv)
    if args.trained_tabletop and not args.yolo_place:
        parser.error("--trained-tabletop requires --yolo-place")
    if args.show_validation and not (args.validate and args.trained_tabletop):
        parser.error("--show-validation requires --validate --trained-tabletop")
    return args


def main():
    args = parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "ros2_ws/.bottle_pick.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_stem = ("bottle_tabletop_place" if args.trained_tabletop else
                       "bottle_yolo_place" if args.yolo_place else "bottle_pick")
        report_path = REPORTS / f"{report_stem}_{'validation' if args.validate else 'execution'}.json"
        if args.trained_tabletop:
            # A failed new attempt must not leave an older successful report in place.
            report_path.write_text(json.dumps({"completed": False, "mode": "validation" if args.validate else "visible_execution",
                                               "detector": TABLETOP_DETECTOR, "error": "Preflight not completed"}) + "\n")
            tabletop_model_hash()
        interpreter = str(ROOT / "env_yolov9_cpu/bin/python") if args.yolo_place else sys.executable
        command = [interpreter, str(ROOT / "scripts/detect_bottle_rgbd.py")]
        if args.trained_tabletop:
            command = [interpreter, str(ROOT / "scripts/wrist_yolo_ros.py"), "--capture"]
        elif args.yolo_place:
            command += ["--detector", "official-yolov9"]
        capture_started = time.time()
        subprocess.run(command, check=True, cwd=ROOT, timeout=90)
        detection_name = ("tabletop_detection.json" if args.trained_tabletop else
                          "bottle_yolov9_detection.json" if args.yolo_place else "bottle_detection.json")
        detection = json.loads((REPORTS / detection_name).read_text())
        require_fresh_detection(detection, not_before=capture_started if args.trained_tabletop else None)
        if args.trained_tabletop:
            validate_tabletop_detection(detection)
        domain = 46 if args.yolo_place else 45
        os.environ["ROS_DOMAIN_ID"] = str(domain)
        os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        from isaacsim import SimulationApp
        visible = args.execute or args.show_validation
        app = SimulationApp({"headless": not visible})
        node = None
        live_camera = None
        started = False
        failed = False
        digest = None
        try:
            import carb
            import omni.timeline
            import omni.usd
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.ros2.bridge")
            app.update()
            import rclpy
            from bottle_pick_controller import BottlePickNode
            from yolo_bottle_place_controller import YOLOBottlePlaceNode
            context = omni.usd.get_context()
            if not context.open_stage(str(SCENE)):
                raise RuntimeError("Could not open clutter scene")
            for _ in range(20):
                app.update()
            stage = context.get_stage()
            digest = fingerprint(stage, trained_tabletop=args.trained_tabletop)
            if args.execute:
                validation = json.loads((REPORTS / f"{report_stem}_validation.json").read_text())
                require_execution_approval(validation, detection, digest, args.trained_tabletop)
            require_fresh_detection(detection, max_age=120)
            stage.SetEditTarget(stage.GetSessionLayer())
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(36000)
            stage.GetPrimAtPath("/World/ROS2").SetActive(False)
            omni.timeline.get_timeline_interface().set_looping(False)
            if visible:
                from isaacsim.core.utils.viewports import set_camera_view
                label = "VISIBLE GUARDED VALIDATION" if args.show_validation else "VALIDATED EXECUTION"
                carb.settings.get_settings().set("/app/window/title", f"ROBOT_VLA — {label} (domain {domain})")
                set_camera_view(eye=[1.7, 1.3, 1.1], target=[.55, .15, .18])
            if args.trained_tabletop:
                from live_motion_camera import LiveMotionCamera
                live_camera = LiveMotionCamera(stage)
            rclpy.init(args=[])
            started = True
            controller = YOLOBottlePlaceNode if args.yolo_place else BottlePickNode
            node = controller(app, URDF, detection)
            if live_camera is not None:
                live_camera.bind(node)
                node.live_camera = live_camera
                warmup_deadline = time.monotonic() + 25
                while live_camera.frames_published < 3:
                    if time.monotonic() > warmup_deadline:
                        raise RuntimeError("Moving wrist RGB failed to initialize; no task motion started")
                    node.tick()
                print("LIVE_MOTION_CAMERA_READY: domain46, same-scene RGB; starting guarded motion", flush=True)
            report = node.run()
            report.update(fingerprint=digest, mode="validation" if args.validate else "visible_execution",
                          vision_capture_time=detection["captured_at_unix"], visible=visible,
                          moving_camera_frames=live_camera.frames_published if live_camera else 0)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            marker = report_stem.upper()
            print(marker + ("_VALIDATED" if args.validate else "_EXECUTED"), flush=True)
        except BaseException as error:
            failed = True
            traceback.print_exc()
            report = {"completed": False, "error": str(error), "fingerprint": digest,
                      "mode": "validation" if args.validate else "visible_execution", "visible": visible,
                      "moving_camera_frames": live_camera.frames_published if live_camera else 0}
            if args.trained_tabletop:
                report.update(detector=TABLETOP_DETECTOR, weights_sha256=detection["weights_sha256"],
                              detected_tray={"center_xy": detection["tray_center_xy"],
                                             "floor_z": detection["tray_floor_z"], "rim_z": detection["tray_rim_z"]})
            if node is not None:
                report.update(phase=node.phase, grasp_xyz=node.grasp().tolist(),
                              finger_positions=node.q()[node.fi].tolist(), events=node.events)
                if args.yolo_place:
                    report.update(grip_diagnostics=node.grip_diagnostics,
                                  bottle_xyz=node.pose("/World/TabletopProps/WaterBottle")[0].tolist())
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print("BOTTLE_PICK_ABORTED: no retry or automatic release", flush=True)
        finally:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            timeline.commit()
            if visible:
                # Keep the result visible, but forbid accidental Play/retry.
                result_director = omni.timeline.get_timeline_interface("bottle_result_hold")
                result_director.pause()
                result_director.set_auto_update(False)
                result_director.commit()
                timeline.set_director(result_director)
                timeline.commit()
                carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — PAUSED: " + ("VALIDATION FAILED — see report" if failed else "TASK COMPLETED"))
                if node is not None:
                    node.emit("aborted" if failed else "completed", result_report=str(report_path))
                while app.is_running():
                    app.update()
                    if started and node is not None:
                        rclpy.spin_once(node, timeout_sec=0.)
            if live_camera is not None:
                live_camera.close()
            if node is not None:
                node.subscription = None
                node.destroy_node()
            if started:
                rclpy.shutdown()
            app.close(skip_cleanup=False)
        return int(failed)


if __name__ == "__main__":
    sys.exit(main())