"""Explicit visible validation of a reconstructed, occupied-box simulation.

Original USD and completed bottle window stay untouched. No hot loading,
post-play pose writes, automatic retry, or release on failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "ros2_ws/src/vla_sim")]
SCENE = ROOT / "isaac/scenes/panda_tabletop_cluttered.usda"
URDF = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
REPORTS = ROOT / "ros2_ws/log"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--show-validation", action="store_true")
    args = parser.parse_args()
    if args.show_validation and not args.validate:
        parser.error("--show-validation requires --validate")
    source_path = REPORTS / "bottle_tabletop_place_validation.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if not all(source.get(k) is True for k in (
            "completed", "returned_home", "box_bottom_contact", "fingers_open")):
        raise RuntimeError("Completed supported bottle placement required for reconstruction")
    from src.control.green_cube_plan import BOTTLE, placement_geometry
    bottle = source["bottle_final_xyz"]
    placement_geometry(bottle)
    os.environ["ROS_DOMAIN_ID"] = "47"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    import fcntl
    lock = (ROOT / "ros2_ws/.green_cube.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    visible = args.execute or args.show_validation
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not visible})
    node = None
    started = False
    failed = True
    digest = None
    path = REPORTS / ("green_cube_validation.json" if args.validate else "green_cube_execution.json")
    report = {"completed": False, "mode": "validation" if args.validate else "execution",
              "visible": visible, "reconstructed_scene": True, "ros_domain": 47,
              "source_bottle_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "initial_bottle_xyz": bottle, "targeting": "simulator-known fixture, not YOLO"}
    try:
        import carb
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        app.update()
        import rclpy
        from green_cube_controller import GreenCubeNode
        context = omni.usd.get_context()
        if not context.open_stage(str(SCENE)):
            raise RuntimeError("Cannot open cluttered scene")
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_stopped():
            raise RuntimeError("Reconstruction requires stopped physics")
        h = hashlib.sha256(source_bytes)
        for relative in ("isaac/scripts/run_green_cube.py", "isaac/scripts/green_cube_controller.py",
                         "isaac/scripts/pick_place_controller.py", "src/control/green_cube_plan.py",
                         "src/control/bottle_grasp_plan.py", "src/control/contact_evidence.py",
                         "ros2_ws/src/vla_sim/vla_sim/motion.py", "ros2_ws/src/vla_sim/vla_sim/kinematics.py"):
            h.update((ROOT / relative).read_bytes())
        h.update(URDF.read_bytes())
        for layer in sorted(stage.GetUsedLayers(), key=lambda x: x.identifier):
            if layer != stage.GetSessionLayer():
                h.update(layer.ExportToString().encode())
        digest = h.hexdigest()
        if args.execute:
            validation = json.loads((REPORTS / "green_cube_validation.json").read_text())
            if not validation.get("completed") or validation.get("fingerprint") != digest:
                raise RuntimeError("Matching completed green validation required")
        # Explicitly authorized reconstructed INITIAL state, before physics.
        # Only translation changes; the source asset's upright rotation remains.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.XformCommonAPI(stage.GetPrimAtPath(BOTTLE)).SetTranslate(Gf.Vec3d(*bottle))
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(36000)
        timeline.set_looping(False)
        if visible:
            carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — GREEN CUBE: RECONSTRUCTED VALIDATION (domain 47)")
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera("/World/Overview")
        rclpy.init(args=[])
        started = True
        node = GreenCubeNode(app, URDF, bottle)
        report.update(node.run())
        failed = False
        print("GREEN_CUBE_VALIDATED" if args.validate else "GREEN_CUBE_EXECUTED", flush=True)
    except BaseException as error:
        traceback.print_exc()
        report.update(completed=False, error=str(error))
        if node is not None:
            report.update(phase=node.phase, events=node.events)
        print("GREEN_CUBE_ABORTED: no retry, no automatic release", flush=True)
    finally:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.commit()
        report["fingerprint"] = digest
        path.write_text(json.dumps(report, indent=2) + "\n")
        if visible:
            director = omni.timeline.get_timeline_interface("green_result_hold")
            director.pause()
            director.set_auto_update(False)
            director.commit()
            timeline.set_director(director)
            timeline.commit()
            import carb
            carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — GREEN CUBE: " + (
                "PAUSED — VALIDATION FAILED" if failed else "COMPLETED — BOTTLE PRESERVED"))
            if node is not None:
                node.emit("aborted" if failed else "completed", result_report=str(path))
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
        lock.close()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())