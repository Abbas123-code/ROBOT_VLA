"""Run in the playing Isaac Script Editor AFTER successful closure. Lift 2 cm.

Keeps finger drives unchanged. Uses live cube/hand poses to detect slip, pauses
physics on failure, and leaves the arm holding after success. Never attaches
the cube or saves the stage. Do not run alongside another motion controller.
"""
import asyncio
import json
import time

import carb
import numpy as np
import omni.kit.app
import omni.physics.tensors as tensors
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.xforms import get_world_pose
from omni.physx import get_physx_simulation_interface
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

START = np.array([-0.00010444936969574028, 0.6344829882310178, 0.0001,
                  -2.0039364326175164, -0.00012292646526497329, 2.6384194152712297, 0.7854819424277798])
TARGET = np.array([-0.000102559339799891, 0.5888807609816573, 0.0001,
                   -2.016574825227355, -0.00010873336423903744, 2.6054555813050606, 0.7854722371549531])
HIGHER_TARGET = np.array([-9.876246313352277e-05, 0.4813436671734749, 0.0001,
                          -2.0345174206912024, -7.904696807635922e-05, 2.5158610843311386, 0.7854521086264532])


async def lift_cube(*, higher=False):
    """First lift by default; higher=True continues from it by another 5 cm."""
    start_target = TARGET.copy() if higher else START.copy()
    end_target = HIGHER_TARGET.copy() if higher else TARGET.copy()
    expected_rise = 0.05 if higher else 0.02
    initial_cube_height = 0.045 if higher else 0.025
    duration = 10.0 if higher else 6.0
    context = omni.usd.get_context()
    stage = context.get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or not timeline.is_playing():
        raise RuntimeError("Open the grasped-cube scene and press Play")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-8:
        raise RuntimeError("Metre stage units required")
    stage_id = context.get_stage_id()
    fabric = usdrt.Usd.Stage.Attach(stage_id)
    paths = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/panda")):
        if prim.IsA(UsdGeom.Xform) and prim.GetName() in ("panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
            if prim.GetName() in paths:
                raise RuntimeError("Ambiguous robot links")
            paths[prim.GetName()] = str(prim.GetPath())
    if len(paths) != 4:
        raise RuntimeError("Missing Panda links")
    # Reuse closure's reporters. Adding new schemas to ancestor links during
    # playback rebuilds this imported articulation and can destroy its DOF view.
    for path in (paths["panda_leftfinger"], paths["panda_rightfinger"], "/World/Cube"):
        if not stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise RuntimeError("Run the tested closure first to enable finger/cube contact reports")

    def pose(path):
        prim = fabric.GetPrimAtPath(path)
        has_fabric = bool(prim) and usdrt.Rt.Xformable(prim).HasWorldXform()
        if not has_fabric and not carb.settings.get_settings().get("/physics/updateToUsd"):
            raise RuntimeError(f"Verified live pose missing: {path}")
        p, q = get_world_pose(path, fabric=has_fabric)
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise RuntimeError("Invalid live pose")
        return np.asarray(p).copy(), np.asarray(q).copy()

    def grasp_pose():
        hand, q = pose(paths["panda_hand"])
        rot = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist())))
        return hand + np.array(rot.TransformDir(Gf.Vec3d(0, 0, 0.105))), q

    view = tensors.create_simulation_view("numpy", stage_id, backend="physx")
    arm = view.create_articulation_view(paths["panda_link0"])
    if arm.count != 1:
        raise RuntimeError("Expected one Panda")
    names = list(arm.shared_metatype.dof_names)
    ai = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    fi = [names.index(f"panda_finger_joint{i}") for i in (1, 2)]
    q0 = arm.get_dof_positions()[0].copy()
    if not np.isfinite(q0).all() or np.max(np.abs(q0[ai] - start_target)) > 0.008:
        raise RuntimeError("Arm is not at the validated starting position for this lift")
    if not np.allclose(q0[fi], [0.02508, 0.02508], atol=0.001):
        raise RuntimeError("Expected fingers gripping the 5 cm cube")
    for i in (1, 2):
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"/World/panda/Physics/panda_finger_joint{i}"), "linear")
        actual = [drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(),
                  drive.GetMaxForceAttr().Get(), drive.GetTargetPositionAttr().Get()]
        if drive.GetTypeAttr().Get() != "force" or not np.allclose(actual, [1000, 20, 5, 0.023], atol=1e-5):
            raise RuntimeError("Expected tested 5 N gripper hold; run closure first")
    base, base_q = pose(paths["panda_link0"])
    if np.linalg.norm(base - [-0.0991988853, -0.0000007233, 0.0000269338]) > 0.002 or abs(base_q[0]) < 0.99999:
        raise RuntimeError("Robot base changed from validated scene")
    grasp0, hand_q0 = grasp_pose()
    cube0, cube_q0 = pose("/World/Cube")
    if np.linalg.norm(grasp0[:2] - cube0[:2]) > 0.006 or abs(grasp0[2] - cube0[2]) > 0.008 or abs(cube0[2] - initial_cube_height) > 0.002:
        raise RuntimeError("Cube is not aligned or at the expected height for this lift")
    drives = [UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"/World/panda/Physics/panda_joint{i}"), "angular") for i in range(1, 8)]
    unexpected = set()
    finger_paths = {paths["panda_leftfinger"], paths["panda_rightfinger"]}

    def contacts(headers, data):
        for header in headers:
            if header.stage_id != stage_id or not header.num_contact_data:
                continue
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(a)) for a in (header.actor0, header.actor1))
            robots = [p for p in pair if p.startswith("/World/panda/")]
            if robots:
                allowed = (len(robots) == 1 and
                           ((robots[0] in finger_paths and "/World/Cube" in pair) or
                            (robots[0] == paths["panda_link0"] and any(p.startswith("/World/table") for p in pair))))
                if not allowed:
                    unexpected.add(pair)

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    start = last = timeline.get_current_time()
    progress = time.monotonic()
    deadline = progress + 45
    peak_slip = peak_tracking = 0.0
    commanded = start_target.copy()
    try:
        while time.monotonic() < deadline:
            await omni.kit.app.get_app().next_update_async()
            if context.get_stage_id() != stage_id or not timeline.is_playing():
                raise RuntimeError("Stage changed or simulation stopped/paused")
            now = timeline.get_current_time()
            if now < last or now - last > 0.25:
                raise RuntimeError("Simulation clock jumped")
            if now > last:
                progress = time.monotonic()
            if time.monotonic() - progress > 1:
                raise RuntimeError("Simulation clock stalled")
            last = now
            q = arm.get_dof_positions()[0].copy()
            tracking = float(np.max(np.abs(q[ai] - commanded)))
            peak_tracking = max(peak_tracking, tracking)
            if not np.isfinite(q).all() or tracking > 0.02 or not np.allclose(q[fi], q0[fi], atol=0.0015):
                raise RuntimeError("Joint tracking or finger opening changed beyond limits")
            grasp, hand_q = grasp_pose()
            cube, cube_q = pose("/World/Cube")
            slip = float(np.linalg.norm((cube - grasp) - (cube0 - grasp0)))
            peak_slip = max(peak_slip, slip)
            if slip > 0.003:
                raise RuntimeError("Cube slipped more than 3 mm relative to gripper")
            if np.linalg.norm(cube[:2] - cube0[:2]) > 0.004 or not -0.001 <= cube[2] - cube0[2] <= expected_rise + 0.005:
                raise RuntimeError("Cube left the small vertical-lift corridor")
            if min(abs(float(np.dot(cube_q, cube_q0))), abs(float(np.dot(hand_q, hand_q0)))) < np.cos(0.08 / 2):
                raise RuntimeError("Cube or gripper tilted")
            if unexpected:
                raise RuntimeError(f"Unexpected robot contacts: {unexpected}")
            elapsed = now - start
            fraction = min(max(elapsed / duration, 0), 1)
            blend = fraction * fraction * (3 - 2 * fraction)
            commanded = start_target + blend * (end_target - start_target)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for drive, angle in zip(drives, commanded):
                    drive.GetTargetPositionAttr().Set(float(np.degrees(angle)))
            if elapsed >= duration + 2:
                rise = float(cube[2] - cube0[2])
                if abs(rise - expected_rise) > 0.004:
                    raise RuntimeError(f"Cube did not complete the expected lift: {rise}")
                report = {"lift_verified": True, "cube_rise_metres": rise,
                          "lift_profile": "additional_5cm" if higher else "initial_2cm",
                          "grasp_rise_metres": float(grasp[2] - grasp0[2]),
                          "maximum_relative_slip_metres": peak_slip,
                          "peak_joint_tracking_error_rad": peak_tracking,
                          "cube_world_xyz": cube.tolist(), "finger_positions_metres": q[fi].tolist(),
                          "note": f"{duration:g} second lift plus 2 second hold, unchanged 5 N finger drives, no artificial attachment. Monitoring ends here."}
                print(json.dumps(report, indent=2), flush=True)
                return report
        raise RuntimeError("Lift exceeded wall-time deadline")
    except BaseException:
        # Freeze simulation on failure; do not open fingers or continue lifting.
        if context.get_stage_id() == stage_id and not timeline.is_stopped():
            timeline.pause()
        raise
    finally:
        subscription = None


if __name__ == "__main__":
    if "panda_lift_task" in globals() and not panda_lift_task.done():
        raise RuntimeError("A lift is already running")
    panda_lift_task = asyncio.ensure_future(lift_cube())

    def report_lift(task):
        try:
            task.result()
        except BaseException as error:
            print(f"LIFT ABORTED (simulation paused if motion started): {error}", flush=True)

    panda_lift_task.add_done_callback(report_lift)