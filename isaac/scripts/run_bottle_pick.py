"""Validate or execute an image-targeted bottle pick in a separate simulation.

Gets a fresh ROS image estimate from the domain-44 frozen preview before launch.
The control simulation uses domain45; no command is sent to an existing scene.
"""
import argparse
import fcntl
import hashlib
import json
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


def fingerprint(stage):
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("bottle_pick_controller.py"),
                 Path(__file__).with_name("yolo_bottle_place_controller.py"),
                 Path(__file__).with_name("pick_place_controller.py"), URDF,
                 ROOT / "src/perception/bottle_rgbd.py", ROOT / "scripts/detect_bottle_rgbd.py",
                 ROOT / "src/perception/yolov9_official.py", ROOT / "configs/yolov9_official.json",
                 ROOT / "src/control/bottle_grasp_plan.py",
                 ROOT / "ros2_ws/src/vla_sim/vla_sim/kinematics.py", ROOT / "ros2_ws/src/vla_sim/vla_sim/motion.py"):
        digest.update(path.read_bytes())
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.identifier):
        if layer != stage.GetSessionLayer():
            digest.update(layer.ExportToString().encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--yolo-place", action="store_true", help="Official YOLOv9 body grasp and box placement (domain46)")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "ros2_ws/.bottle_pick.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        interpreter = str(ROOT / "env_yolov9_cpu/bin/python") if args.yolo_place else sys.executable
        command = [interpreter, str(ROOT / "scripts/detect_bottle_rgbd.py")]
        if args.yolo_place:
            command += ["--detector", "official-yolov9"]
        subprocess.run(command, check=True, cwd=ROOT, timeout=90)
        detection = json.loads((REPORTS / ("bottle_yolov9_detection.json" if args.yolo_place else "bottle_detection.json")).read_text())
        if not detection["completed"] or time.time() - detection["captured_at_unix"] > 15:
            raise RuntimeError("Fresh vision estimate required")
        domain = 46 if args.yolo_place else 45
        os.environ["ROS_DOMAIN_ID"] = str(domain)
        os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.validate})
        node = None
        started = False
        failed = False
        digest = None
        report_stem = "bottle_yolo_place" if args.yolo_place else "bottle_pick"
        report_path = REPORTS / f"{report_stem}_{'validation' if args.validate else 'execution'}.json"
        try:
            import carb
            import numpy as np
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
            digest = fingerprint(stage)
            if args.execute:
                validation = json.loads((REPORTS / f"{report_stem}_validation.json").read_text())
                if not validation.get("completed") or validation.get("fingerprint") != digest:
                    raise RuntimeError("Current successful isolated validation required")
                if np.linalg.norm(np.asarray(validation["vision_target_xy"]) - detection["center_xy"]) > .002:
                    raise RuntimeError("Image target changed since isolated validation")
            if time.time() - detection["captured_at_unix"] > 120:
                raise RuntimeError("Vision snapshot became stale during startup")
            stage.SetEditTarget(stage.GetSessionLayer())
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(36000)
            stage.GetPrimAtPath("/World/ROS2").SetActive(False)
            omni.timeline.get_timeline_interface().set_looping(False)
            if args.execute:
                from isaacsim.core.utils.viewports import set_camera_view
                carb.settings.get_settings().set("/app/window/title", f"ROBOT_VLA — {'YOLO bottle placement' if args.yolo_place else 'Bottle pickup'} (domain {domain})")
                set_camera_view(eye=[1.7, 1.3, 1.1], target=[.55, .15, .18])
            rclpy.init(args=[])
            started = True
            controller = YOLOBottlePlaceNode if args.yolo_place else BottlePickNode
            node = controller(app, URDF, detection)
            report = node.run()
            report.update(fingerprint=digest, mode="validation" if args.validate else "visible_execution",
                          vision_capture_time=detection["captured_at_unix"])
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            marker = "BOTTLE_YOLO_PLACE" if args.yolo_place else "BOTTLE_PICK"
            print(marker + ("_VALIDATED" if args.validate else "_EXECUTED"), flush=True)
        except BaseException as error:
            failed = True
            traceback.print_exc()
            report = {"completed": False, "error": str(error), "fingerprint": digest}
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
            omni.timeline.get_timeline_interface().pause()
            if args.execute:
                while app.is_running():
                    app.update()
                    if started and node is not None:
                        rclpy.spin_once(node, timeout_sec=0.)
            if node is not None:
                node.subscription = None
                node.destroy_node()
            if started:
                rclpy.shutdown()
            app.close(skip_cleanup=False)
        return int(failed)


if __name__ == "__main__":
    sys.exit(main())