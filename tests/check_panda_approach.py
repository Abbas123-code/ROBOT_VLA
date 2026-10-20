import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/vla_sim"))
from vla_sim.approach import check_grasp_position, load_plan, sample
from vla_sim.motion import JOINT_NAMES, ordered_positions

parser = argparse.ArgumentParser(description="Validate an open-finger motion plan in an isolated scene")
parser.add_argument("--plan", type=Path, default=ROOT / "configs/panda_approach.json")
args = parser.parse_args()
PLAN, PLAN_HASH = load_plan(args.plan)
SCENE = ROOT / "isaac/scenes/panda_tabletop.usda"
REPORT = ROOT / "ros2_ws/log" / f"{args.plan.stem}_validation.json"

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.physics.tensors as tensors
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.utils.xforms import get_world_pose
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError("Cannot open test scene")
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    stage.RemovePrim("/World/ROS2")
    robot = stage.GetPrimAtPath("/World/panda")
    UsdGeom.XformCommonAPI(robot).SetTranslate(Gf.Vec3d(*PLAN["base_world_xyz"]))
    np.testing.assert_allclose(PLAN["base_quaternion_wxyz"], [1, 0, 0, 0])
    joints = {}
    root_path = hand_path = None
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_path = str(prim.GetPath())
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
        if prim.GetName() == "panda_hand" and prim.IsA(UsdGeom.Xform):
            hand_path = str(prim.GetPath())
        if prim.GetName() in JOINT_NAMES and prim.IsA(UsdPhysics.RevoluteJoint):
            joints[prim.GetName()] = prim
            angle = PLAN["start_arm_radians"][JOINT_NAMES.index(prim.GetName())]
            PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr(np.degrees(angle))
            UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(np.degrees(angle))
        if prim.IsA(UsdPhysics.PrismaticJoint):
            PhysxSchema.JointStateAPI.Apply(prim, "linear").CreatePositionAttr(0.04)
    assert root_path and hand_path and len(joints) == 7
    fixed_groups = {}

    def group(path):
        fixed_groups.setdefault(path, path)
        while fixed_groups[path] != path:
            path = fixed_groups[path]
        return path

    for prim in Usd.PrimRange(robot):
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            bodies = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if bodies[0] and bodies[1]:
                a, b = str(bodies[0][0]), str(bodies[1][0])
                fixed_groups[group(a)] = group(b)
    fixed_pairs = []
    for a in list(fixed_groups):
        for b in list(fixed_groups):
            if a < b and group(a) == group(b):
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(b)
                fixed_pairs.append((a, b))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    contacts, forbidden = set(), set()

    def on_contacts(headers, data):
        for header in headers:
            if header.stage_id != context.get_stage_id() or header.num_contact_data == 0:
                continue
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(actor))
                         for actor in (header.actor0, header.actor1))
            contacts.add(pair)
            robot_actors = [path for path in pair if path.startswith("/World/panda/")]
            if len(robot_actors) == 2:
                forbidden.add(pair)
            elif robot_actors:
                other = next(path for path in pair if path != robot_actors[0])
                if not (robot_actors[0] == root_path and other.startswith("/World/table")):
                    forbidden.add(pair)

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
    view = tensors.create_simulation_view("numpy", context.get_stage_id(), backend="physx")
    arm = view.create_articulation_view(root_path)
    assert arm.count == 1
    indices = [list(arm.shared_metatype.dof_names).index(name) for name in JOINT_NAMES]
    base, _ = get_world_pose(root_path, fabric=True)
    np.testing.assert_allclose(base, PLAN["base_world_xyz"], atol=1e-3)
    initial = arm.get_dof_positions()[0, indices].copy()
    np.testing.assert_allclose(initial, PLAN["start_arm_radians"], atol=0.015)
    peak_tracking, min_grasp_height, max_lateral_offset = 0.0, float("inf"), 0.0
    start = timeline.get_current_time()
    last_publish = -float("inf")
    target = PLAN["start_arm_radians"]
    complete = False
    for _ in range(2400):
        elapsed = timeline.get_current_time() - start
        if elapsed - last_publish >= 0.05:
            target = sample(PLAN, elapsed)
            for name, radians in zip(JOINT_NAMES, target):
                UsdPhysics.DriveAPI(joints[name], "angular").GetTargetPositionAttr().Set(float(np.degrees(radians)))
            last_publish = elapsed
        app.update()
        actual = arm.get_dof_positions()[0, indices].copy()
        ordered_positions(JOINT_NAMES, actual)
        tracking = float(np.max(np.abs(actual - target)))
        peak_tracking = max(peak_tracking, tracking)
        assert np.isfinite(actual).all() and tracking < PLAN["max_joint_tracking_error_rad"], actual
        assert not forbidden, f"Unexpected contacts: {forbidden}"
        hand, q = get_world_pose(hand_path, fabric=True)
        rotation = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist())))
        grasp = np.asarray(hand) + np.array(rotation.TransformDir(Gf.Vec3d(0, 0, 0.105)))
        check_grasp_position(PLAN, grasp)
        min_grasp_height = min(min_grasp_height, float(grasp[2]))
        max_lateral_offset = max(max_lateral_offset, float(np.linalg.norm(grasp[:2] - np.asarray(PLAN["target_world_xyz"])[:2])))
        if elapsed >= PLAN["duration_seconds"] + PLAN["settle_seconds"]:
            complete = True
            break
    assert complete, "Test did not reach its final simulated time"
    assert contacts, "Contact reporting was not exercised"
    assert any("/World/Cube" in pair for pair in contacts), contacts
    cube, _ = get_world_pose("/World/Cube", fabric=True)
    np.testing.assert_allclose(cube, PLAN["cube_world_xyz"], atol=0.002)
    position_error = float(np.linalg.norm(grasp - PLAN["target_world_xyz"]))
    assert position_error < PLAN["max_final_position_error_metres"], position_error
    orientation_error = float(np.linalg.norm(rotation.TransformDir(Gf.Vec3d(0, 0, 1)) - Gf.Vec3d(0, 0, -1)))
    assert orientation_error < 0.03, orientation_error
    report = {
        "passed": True, "plan_sha256": PLAN_HASH,
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "source": "separate unsaved scene, ROS removed, base offset reproduced, self collisions enabled",
        "base_xyz": np.asarray(base).tolist(), "final_grasp_world_xyz": grasp.tolist(),
        "final_arm_radians": actual.tolist(), "position_error_metres": position_error,
        "peak_tracking_error_rad": peak_tracking, "minimum_grasp_height_metres": min_grasp_height,
        "maximum_lateral_offset_metres": max_lateral_offset,
        "observed_contact_pairs": sorted(contacts), "unexpected_contacts": sorted(forbidden),
        "excluded_fixed_joint_pairs": fixed_pairs,
        "scope": "Finite-step PhysX check of this imported geometry, not a continuous collision-free proof or live-scene monitor",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    timeline.stop()
    for _ in range(10):
        app.update()
except Exception:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"passed": False, "plan_sha256": PLAN_HASH}) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close(skip_cleanup=False)