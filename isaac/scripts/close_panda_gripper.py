"""Run in Isaac Sim Script Editor while playing: close gently, hold, never lift.

Edits only the session layer. Successful drive targets remain active after the
task ends; this is a motor hold, not a fixed constraint or verified lifting grasp.
"""

import asyncio
import json
import time

import numpy as np
import omni.kit.app
import omni.physics.tensors as tensors
import omni.timeline
import omni.usd
from isaacsim.core.utils.xforms import get_world_pose
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics


async def close_gripper():
    context = omni.usd.get_context()
    stage = context.get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or not timeline.is_playing():
        raise RuntimeError("Open the Panda scene and press Play first")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-8:
        raise RuntimeError("This controller requires metre stage units")
    stage_id = context.get_stage_id()
    robot = stage.GetPrimAtPath("/World/panda")
    if not robot.IsValid():
        raise RuntimeError("Panda is missing")
    paths = {}
    for prim in Usd.PrimRange(robot):
        if prim.GetName() in ("panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger") and prim.IsA(UsdGeom.Xform):
            if prim.GetName() in paths:
                raise RuntimeError("Ambiguous robot link")
            paths[prim.GetName()] = str(prim.GetPath())
    if len(paths) != 4:
        raise RuntimeError("Missing gripper links")
    view = tensors.create_simulation_view("numpy", stage_id, backend="physx")
    arm = view.create_articulation_view(paths["panda_link0"])
    if arm.count != 1:
        raise RuntimeError("Expected one Panda articulation")
    names = list(arm.shared_metatype.dof_names)
    ai = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    fi = [names.index(f"panda_finger_joint{i}") for i in (1, 2)]
    initial = arm.get_dof_positions()[0].copy()
    if not np.isfinite(initial).all() or not np.allclose(initial[fi], [0.04, 0.04], atol=0.001):
        raise RuntimeError("Expected both fingers open at 0.04 m")
    hand, quat = get_world_pose(paths["panda_hand"], fabric=True)
    rotation = Gf.Rotation(Gf.Quatd(float(quat[0]), Gf.Vec3d(*np.asarray(quat[1:]).tolist())))
    grasp = np.asarray(hand) + np.array(rotation.TransformDir(Gf.Vec3d(0, 0, 0.105)))
    cube, cube_quat = get_world_pose("/World/Cube", fabric=True)
    cube = np.asarray(cube).copy()
    cube_quat = np.asarray(cube_quat).copy()
    if np.linalg.norm(grasp[:2] - cube[:2]) > 0.006 or not -0.005 <= grasp[2] - cube[2] <= 0.008:
        raise RuntimeError("Live gripper is not aligned with the cube centre")
    if np.linalg.norm(np.array(rotation.TransformDir(Gf.Vec3d(0, 0, 1))) - [0, 0, -1]) > 0.03:
        raise RuntimeError("Gripper is not pointing down")
    drives = [UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"/World/panda/Physics/panda_finger_joint{i}"), "linear") for i in (1, 2)]
    fingers = {paths["panda_leftfinger"], paths["panda_rightfinger"]}
    active_contacts, unexpected = set(), set()

    def contacts(headers, data):
        for header in headers:
            if header.stage_id != stage_id:
                continue
            pair = [str(PhysicsSchemaTools.intToSdfPath(a)) for a in (header.actor0, header.actor1)]
            colliders = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(c)) for c in (header.collider0, header.collider1)))
            for finger in fingers.intersection(pair):
                other = pair[1] if pair[0] == finger else pair[0]
                if other == "/World/Cube":
                    key = (finger, colliders)
                    if header.type == ContactEventType.CONTACT_LOST:
                        active_contacts.discard(key)
                    elif header.num_contact_data:
                        active_contacts.add(key)
                elif header.num_contact_data:
                    unexpected.add(tuple(pair))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    configured = False
    start = timeline.get_current_time()
    last_time, last_progress = start, time.monotonic()
    deadline = time.monotonic() + 30
    bilateral_since = None
    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in fingers | {"/World/Cube"}:
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
            for drive, position in zip(drives, initial[fi]):
                drive.CreateTypeAttr("force")
                drive.CreateMaxForceAttr(5.0)
                drive.CreateStiffnessAttr(1000.0)
                drive.CreateDampingAttr(20.0)
                drive.CreateTargetPositionAttr(float(position))
                drive.CreateTargetVelocityAttr(0.0)
        configured = True
        while time.monotonic() < deadline:
            await omni.kit.app.get_app().next_update_async()
            if context.get_stage_id() != stage_id or not timeline.is_playing():
                raise RuntimeError("Stage changed or simulation paused/stopped")
            now = timeline.get_current_time()
            if now < last_time or now - last_time > 0.25:
                raise RuntimeError("Simulation clock jumped")
            if now > last_time:
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > 1:
                raise RuntimeError("Simulation clock stalled")
            last_time = now
            q = arm.get_dof_positions()[0].copy()
            if not np.isfinite(q).all() or np.max(np.abs(q[ai] - initial[ai])) > 0.01:
                raise RuntimeError("Arm moved during finger-only closure")
            if np.any(q[fi] < 0.021) or np.any(q[fi] > 0.041) or abs(q[fi][0] - q[fi][1]) > 0.004:
                raise RuntimeError("Finger position or symmetry exceeded limits")
            current_cube, current_quat = get_world_pose("/World/Cube", fabric=True)
            if np.linalg.norm(np.asarray(current_cube) - cube) > 0.005:
                raise RuntimeError("Cube moved more than 5 mm; stopping closure")
            if abs(float(np.dot(cube_quat, np.asarray(current_quat)))) < np.cos(0.1 / 2):
                raise RuntimeError("Cube tilted during closure")
            if unexpected:
                raise RuntimeError(f"Unexpected finger contacts: {unexpected}")
            elapsed = now - start
            fraction = min(max(elapsed / 4.0, 0.0), 1.0)
            blend = fraction * fraction * (3 - 2 * fraction)
            target = 0.04 + (0.023 - 0.04) * blend
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for drive in drives:
                    drive.GetTargetPositionAttr().Set(float(target))
            # Sleeping bodies stop emitting PERSIST; contact remains until LOST.
            bilateral = all(any(key[0] == f for key in active_contacts) for f in fingers)
            bilateral_since = (now if bilateral_since is None else bilateral_since) if bilateral else None
            if elapsed >= 6:
                if bilateral_since is None or now - bilateral_since < 0.5:
                    raise RuntimeError(f"No sustained bilateral contact; fingers={q[fi].tolist()}, active_contacts={active_contacts}; grasp NOT confirmed")
                report = {"closed_and_holding": True, "finger_positions_metres": q[fi].tolist(),
                          "drive_target_metres": target, "drive_force_limit_newtons_per_finger": 5,
                          "bilateral_contact_seconds": now - bilateral_since,
                          "cube_displacement_metres": float(np.linalg.norm(np.asarray(current_cube) - cube)),
                          "arm_max_change_rad": float(np.max(np.abs(q[ai] - initial[ai]))),
                          "note": "Motor hold with confirmed bilateral contact. No lift or fixed attachment; load-bearing grasp not tested."}
                print(json.dumps(report, indent=2), flush=True)
                return report
        raise RuntimeError("Closure timed out")
    except BaseException:
        if configured and context.get_stage_id() == stage_id and timeline.is_playing():
            # Remove squeeze on failure without moving the arm or forcibly reopening.
            q = arm.get_dof_positions()[0].copy()
            if np.isfinite(q[fi]).all():
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    for drive, position in zip(drives, q[fi]):
                        drive.GetTargetPositionAttr().Set(float(position))
        raise
    finally:
        subscription = None


if __name__ == "__main__":
    if "panda_grasp_task" in globals() and not panda_grasp_task.done():
        raise RuntimeError("A gripper closure is already running")
    panda_grasp_task = asyncio.ensure_future(close_gripper())

    def report_result(task):
        try:
            task.result()
        except BaseException as error:
            print(f"GRASP ABORTED: {error}", flush=True)

    panda_grasp_task.add_done_callback(report_result)