"""Exercise the exact GUI closure coroutine in an unsaved ROS-free simulation."""
import asyncio
import argparse
import json
from pathlib import Path
import runpy
import traceback

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lift", action="store_true", help="Test a monitored 2 cm lift after closure")
parser.add_argument("--higher", action="store_true", help="Test initial lift then an additional 5 cm lift")
args = parser.parse_args()
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    plan = json.loads((ROOT / "configs/panda_cube_entry.json").read_text())
    context = omni.usd.get_context()
    assert context.open_stage(str(ROOT / "isaac/scenes/panda_tabletop.usda"))
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    stage.RemovePrim("/World/ROS2")
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/panda")).SetTranslate(Gf.Vec3d(*plan["base_world_xyz"]))
    for i, angle in enumerate(plan["target_arm_radians"], start=1):
        prim = stage.GetPrimAtPath(f"/World/panda/Physics/panda_joint{i}")
        PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr(float(np.degrees(angle)))
        UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(float(np.degrees(angle)))
    for i in (1, 2):
        prim = stage.GetPrimAtPath(f"/World/panda/Physics/panda_finger_joint{i}")
        PhysxSchema.JointStateAPI.Apply(prim, "linear").CreatePositionAttr(0.04)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
    module = runpy.run_path(str(ROOT / "isaac/scripts/close_panda_gripper.py"), run_name="grasp_test")
    task = asyncio.ensure_future(module["close_gripper"]())
    for _ in range(1800):
        app.update()
        if task.done():
            break
    assert task.done(), "Closure coroutine did not complete"
    result = task.result()
    assert result["closed_and_holding"]
    (ROOT / "ros2_ws/log/panda_grasp_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GRASP_VALIDATION_PASSED", flush=True)
    if args.lift or args.higher:
        module = runpy.run_path(str(ROOT / "isaac/scripts/lift_panda_cube.py"), run_name="lift_test")
        task = asyncio.ensure_future(module["lift_cube"]())
        for _ in range(2400):
            app.update()
            if task.done():
                break
        assert task.done(), "Lift coroutine did not complete"
        result = task.result()
        assert result["lift_verified"]
        (ROOT / "ros2_ws/log/panda_lift_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print("LIFT_VALIDATION_PASSED", flush=True)
    if args.higher:
        # Exercise the same entry script the user pastes into the GUI.
        entry = runpy.run_path(str(ROOT / "isaac/scripts/lift_panda_cube_higher.py"), run_name="higher_lift_test")
        task = entry["panda_lift_task"]
        for _ in range(2400):
            app.update()
            if task.done():
                break
        assert task.done(), "Higher lift did not complete"
        result = task.result()
        assert result["lift_verified"] and result["lift_profile"] == "additional_5cm"
        (ROOT / "ros2_ws/log/panda_higher_lift_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print("HIGHER_LIFT_VALIDATION_PASSED", flush=True)
    timeline.stop()
    for _ in range(10):
        app.update()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close(skip_cleanup=False)