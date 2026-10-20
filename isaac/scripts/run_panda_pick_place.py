"""Launch the dedicated ROS-controlled pick/place simulation, not the old GUI.

First --validate (headless). Then --execute (visible, needs matching validation).
The saved scene is never modified. One process owns simulation and ROS control.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws/src/vla_sim"))
SCENE = ROOT / "isaac/scenes/panda_tabletop_with_box.usda"
URDF = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
REPORTS = ROOT / "ros2_ws/log"


def fingerprint(stage):
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "isaac/scripts/pick_place_controller.py", URDF,
                 ROOT / "ros2_ws/src/vla_sim/vla_sim/kinematics.py",
                 ROOT / "ros2_ws/src/vla_sim/vla_sim/motion.py"):
        digest.update(path.read_bytes())
    for layer in sorted(stage.GetUsedLayers(), key=lambda layer: layer.identifier):
        if layer != stage.GetSessionLayer():
            digest.update(layer.ExportToString().encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # Separate domain + namespace: never command another open simulator.
    os.environ["ROS_DOMAIN_ID"] = "43"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    # Prevent concurrent task instances, without touching unrelated GUI sessions.
    import fcntl
    lock = (ROOT / "ros2_ws/.pick_place.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.validate})
    node = None
    ros_started = False
    failed = False
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / ("panda_pick_place_validation.json" if args.validate else "panda_pick_place_execution.json")
    digest = None
    try:
        import carb
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        app.update()
        import rclpy
        from pick_place_controller import PickPlaceNode

        context = omni.usd.get_context()
        if not context.open_stage(str(SCENE)):
            raise RuntimeError("Cannot open box scene")
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        digest = fingerprint(stage)
        if args.execute:
            validation = json.loads((REPORTS / "panda_pick_place_validation.json").read_text())
            if not validation.get("completed") or validation.get("fingerprint") != digest:
                raise RuntimeError("Missing/current validation required before visible execution")
            carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — Automated Pick and Place (domain 43)")
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera("/World/Overview")
        # Layered scene has no authored playback range. Without this, Kit 6.2
        # loops a single frame and rewinds the controller's simulation clock.
        # These edits stay in memory; no layer is saved.
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(36000)
        omni.timeline.get_timeline_interface().set_looping(False)
        rclpy.init(args=[])
        ros_started = True
        node = PickPlaceNode(app, URDF)
        report = node.run()
        report.update(fingerprint=digest, mode="validation" if args.validate else "visible_execution")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        omni.timeline.get_timeline_interface().pause()
        print("PICK_PLACE_VALIDATION_PASSED" if args.validate else "PICK_PLACE_EXECUTION_PASSED", flush=True)
    except BaseException as error:
        failed = True
        traceback.print_exc()
        try:
            import omni.timeline
            omni.timeline.get_timeline_interface().pause()
            if node is not None:
                node.emit("aborted", error=str(error))
        finally:
            report_path.write_text(json.dumps({"completed": False, "fingerprint": digest, "error": str(error)}, indent=2) + "\n")
        print("PICK_PLACE_ABORTED: physics paused; no automatic release/retry", flush=True)
    finally:
        # Leave the visible final/failed state open for inspection, without
        # advancing physics or accepting another task.
        if args.execute:
            while app.is_running():
                app.update()
                if node is not None and ros_started:
                    rclpy.spin_once(node, timeout_sec=0.0)
        if node is not None:
            node.subscription = None
            node.destroy_node()
        if ros_started:
            rclpy.shutdown()
        app.close(skip_cleanup=False)
        lock.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())