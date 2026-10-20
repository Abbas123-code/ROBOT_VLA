"""Inspect the existing bottle controller's solver state at HOME, without picking.

Runs only in a fresh headless simulation. Never saves a source USD layer.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "isaac/scripts"), str(ROOT / "ros2_ws/src/vla_sim")]
os.environ["ROS_DOMAIN_ID"] = "47"


def main():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    node = None
    try:
        import omni.usd
        import omni.timeline
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        app.update()
        import rclpy
        from pxr import Usd, UsdPhysics
        from run_bottle_pick import SCENE, URDF
        from yolo_bottle_place_controller import YOLOBottlePlaceNode
        context = omni.usd.get_context()
        if not context.open_stage(str(SCENE)):
            raise RuntimeError("Cannot open scene")
        for _ in range(20):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath("/World/ROS2").SetActive(False)
        detection = json.loads((ROOT / "ros2_ws/log/bottle_yolov9_detection.json").read_text())
        rclpy.init()
        node = YOLOBottlePlaceNode(app, URDF, detection)
        node.hold(.1)
        report = {"scope": "HOME setup inspection only, not grasp validation",
                  "finger_solver_max_force": node.arm.get_dof_max_forces()[0, node.fi].tolist(),
                  "finger_solver_stiffness": node.arm.get_dof_stiffnesses()[0, node.fi].tolist(),
                  "finger_solver_damping": node.arm.get_dof_dampings()[0, node.fi].tolist(),
                  "finger_usd_max_force": [d.GetMaxForceAttr().Get() for d in node.finger_drives]}
        materials = {}
        for path in [*node.fingers, "/World/TabletopProps/WaterBottle"]:
            view = node.view.create_rigid_body_view(path)
            materials[path] = {"count": view.count, "properties": view.get_material_properties().tolist()}
        report["solver_materials"] = materials
        report["authored_physics_materials"] = [str(p.GetPath()) for p in stage.Traverse()
                                                   if p.HasAPI(UsdPhysics.MaterialAPI)]
        (ROOT / "ros2_ws/log/bottle_grasp_setup.json").write_text(json.dumps(report, indent=2) + "\n")
        print("BOTTLE_SETUP " + json.dumps(report), flush=True)
    finally:
        import omni.timeline
        omni.timeline.get_timeline_interface().pause()
        if node is not None:
            node.subscription = None
            node.destroy_node()
        app.close()


if __name__ == "__main__":
    main()