"""One ROS node coordinating a bounded, simulation-only Panda pick/place task.

Loaded AFTER SimulationApp. Direct Isaac drive control and ROS telemetry share
one owner. The old ROS command graph stays disabled. No attachment or teleport
is used after physics starts. Exceptions are handled by the launcher: pause.
"""
import json
import time

import carb
import numpy as np
import omni.physics.tensors as tensors
import omni.timeline
import omni.usd
import rclpy
import usdrt
from isaacsim.core.utils.xforms import get_world_pose
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from vla_sim.kinematics import link_transform
from vla_sim.motion import HOME, JOINT_NAMES, LIMITS, ordered_positions


class PickPlaceNode(Node):
    def __init__(self, app, urdf):
        super().__init__("panda_pick_place", namespace="vla_pick_place")
        self.app, self.urdf = app, urdf
        self.usd_context = omni.usd.get_context()
        self.stage = self.usd_context.get_stage()
        self.stage_id = self.usd_context.get_stage_id()
        self.timeline = omni.timeline.get_timeline_interface()
        self.phase = "initializing"
        self.abort_requested = False
        self.status = self.create_publisher(String, "status", QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.feedback = self.create_publisher(JointState, "joint_states", 10)
        self.create_service(Trigger, "pause", self.request_pause)
        self.active_contacts, self.unexpected = set(), set()
        self.peak_tracking = self.peak_slip = 0.0
        self.last_publish = -1.0
        self.deadline = time.monotonic() + 600
        self.carry_offset = None
        self.released_cube = None
        self.events = []
        if self.timeline.is_playing():
            raise RuntimeError("Controller requires a fresh stopped stage")
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1) > 1e-8:
            raise RuntimeError("Metre units required")
        graph = self.stage.GetPrimAtPath("/World/ROS2")
        if graph and graph.IsActive():
            raise RuntimeError("Legacy ROS graph must remain disabled")
        robot = self.stage.GetPrimAtPath("/World/panda")
        self.paths = {}
        for prim in Usd.PrimRange(robot):
            if prim.IsA(UsdGeom.Xform) and prim.GetName() in (
                "panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger"
            ):
                if prim.GetName() in self.paths:
                    raise RuntimeError("Ambiguous Panda links")
                self.paths[prim.GetName()] = str(prim.GetPath())
        if len(self.paths) != 4:
            raise RuntimeError("Required Panda links missing")
        self.fingers = {self.paths["panda_leftfinger"], self.paths["panda_rightfinger"]}
        self.arm_drives = [UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(
            f"/World/panda/Physics/panda_joint{i}"), "angular") for i in range(1, 8)]
        self.finger_drives = [UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(
            f"/World/panda/Physics/panda_finger_joint{i}"), "linear") for i in (1, 2)]
        self.arm_target = np.asarray(HOME)
        # All topology/contact schemas are authored BEFORE playback/tensor views.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for prim in self.stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            for i, drive in enumerate(self.arm_drives):
                prim = drive.GetPrim()
                PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr(float(np.degrees(HOME[i])))
                drive.GetTargetPositionAttr().Set(float(np.degrees(HOME[i])))
            for drive in self.finger_drives:
                PhysxSchema.JointStateAPI.Apply(drive.GetPrim(), "linear").CreatePositionAttr(0.04)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(1000.0)
                drive.CreateDampingAttr(20.0)
                drive.CreateMaxForceAttr(5.0)
                drive.CreateTargetPositionAttr(0.04)
                drive.CreateTargetVelocityAttr(0.0)
            self.configure_self_collisions(robot)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.contacts)
        self.timeline.play()
        for _ in range(120):
            self.app.update()
        self.fabric = usdrt.Usd.Stage.Attach(self.stage_id)
        self.view = tensors.create_simulation_view("numpy", self.stage_id, backend="physx")
        self.arm = self.view.create_articulation_view(self.paths["panda_link0"])
        if self.arm.count != 1:
            raise RuntimeError("Expected one Panda")
        self.names = list(self.arm.shared_metatype.dof_names)
        self.ai = [self.names.index(n) for n in JOINT_NAMES]
        self.fi = [self.names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        self.base, base_q = self.pose(self.paths["panda_link0"])
        if np.linalg.norm(self.base) > 0.002 or abs(base_q[0]) < 0.99999:
            raise RuntimeError("Expected baseline zero-offset upright base, not old shifted live scene")
        self.box = np.array([0.45, -0.25])
        box_prim = self.stage.GetPrimAtPath("/World/PlacementBox")
        box_center = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        if not np.allclose(box_center, [*self.box, 0], atol=1e-6):
            raise RuntimeError("Box was moved")
        self.cube_initial, self.cube_quat = self.pose("/World/Cube")
        if not np.allclose(self.cube_initial, [0.5, 0, 0.025], atol=0.002):
            raise RuntimeError("Cube is not at baseline settled position")
        self.last_time = self.timeline.get_current_time()
        self.progress_time = time.monotonic()
        self.emit("ready")

    def configure_self_collisions(self, robot):
        PhysxSchema.PhysxArticulationAPI.Apply(
            self.stage.GetPrimAtPath(self.paths["panda_link0"])
        ).CreateEnabledSelfCollisionsAttr(True)
        groups = {}

        def root(path):
            groups.setdefault(path, path)
            while groups[path] != path:
                path = groups[path]
            return path

        for prim in Usd.PrimRange(robot):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if a and b:
                    groups[root(str(a[0]))] = root(str(b[0]))
        for a in list(groups):
            for b in list(groups):
                if a < b and root(a) == root(b):
                    UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(b)

    def request_pause(self, request, response):
        self.abort_requested = True
        response.success = True
        response.message = "Pause requested; no automatic restart or release"
        return response

    def emit(self, phase, **values):
        self.phase = phase
        record = {"phase": phase, "simulation_time": self.timeline.get_current_time(), **values}
        self.events.append(record)
        self.status.publish(String(data=json.dumps(record)))
        print("PICK_PLACE " + json.dumps(record), flush=True)

    def pose(self, path):
        prim = self.fabric.GetPrimAtPath(path)
        live = bool(prim) and usdrt.Rt.Xformable(prim).HasWorldXform()
        if not live and not carb.settings.get_settings().get("/physics/updateToUsd"):
            raise RuntimeError(f"Live pose unavailable: {path}")
        p, q = get_world_pose(path, fabric=live)
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise RuntimeError("Non-finite physics pose")
        return np.asarray(p).copy(), np.asarray(q).copy()

    def grasp(self):
        hand, q = self.pose(self.paths["panda_hand"])
        rot = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist())))
        return hand + np.asarray(rot.TransformDir(Gf.Vec3d(0, 0, 0.105)))

    def contacts(self, headers, data):
        for header in headers:
            if header.stage_id != self.stage_id:
                continue
            actors = tuple(str(PhysicsSchemaTools.intToSdfPath(a)) for a in (header.actor0, header.actor1))
            colliders = tuple(str(PhysicsSchemaTools.intToSdfPath(c)) for c in (header.collider0, header.collider1))
            key = (tuple(sorted(actors)), tuple(sorted(colliders)))
            if header.type == ContactEventType.CONTACT_LOST:
                self.active_contacts.discard(key)
                continue
            if not header.num_contact_data:
                continue
            self.active_contacts.add(key)
            robots = [a for a in actors if a.startswith("/World/panda/")]
            if robots:
                permitted = len(robots) == 1 and (
                    (robots[0] in self.fingers and "/World/Cube" in actors and self.phase in (
                        "close", "grasp_verified", "lift", "transfer", "lower", "release", "released"
                    )) or (robots[0] == self.paths["panda_link0"] and any(
                        p.startswith("/World/table/") for p in (*actors, *colliders))))
                if not permitted:
                    self.unexpected.add(str((self.phase, actors, colliders)))
            if "/World/Cube" in actors:
                box_contacts = [p for p in (*actors, *colliders) if p.startswith("/World/PlacementBox/")]
                if any("/Bottom" not in p for p in box_contacts):
                    self.unexpected.add("Cube hit box wall: " + str(box_contacts))
                if box_contacts and self.phase not in ("lower", "release", "released", "retreat", "verify", "completed"):
                    self.unexpected.add("Early box contact: " + self.phase)

    def q(self):
        q = self.arm.get_dof_positions()[0].copy()
        ordered_positions(JOINT_NAMES, q[self.ai])
        if not np.isfinite(q).all():
            raise RuntimeError("Non-finite joints")
        return q

    def tick(self):
        if not self.app.is_running() or time.monotonic() > self.deadline:
            raise RuntimeError("App closed or task exceeded 600 s wall-time")
        rclpy.spin_once(self, timeout_sec=0.0)
        if self.abort_requested:
            raise RuntimeError("Operator requested pause")
        self.app.update()
        if self.usd_context.get_stage_id() != self.stage_id or not self.timeline.is_playing():
            raise RuntimeError("Stage or playback changed")
        now = self.timeline.get_current_time()
        if now < self.last_time or now - self.last_time > 0.25:
            raise RuntimeError("Simulation clock jumped")
        if now > self.last_time:
            self.progress_time = time.monotonic()
        if time.monotonic() - self.progress_time > 2:
            raise RuntimeError("Simulation clock stalled")
        self.last_time = now
        q = self.q()
        tracking = float(np.max(np.abs(q[self.ai] - self.arm_target)))
        self.peak_tracking = max(self.peak_tracking, tracking)
        if tracking > 0.04:
            raise RuntimeError(f"Arm tracking exceeded 0.04 rad: {tracking}")
        if self.unexpected:
            raise RuntimeError(f"Unexpected contacts: {sorted(self.unexpected)}")
        cube, quat = self.pose("/World/Cube")
        if abs(float(np.dot(quat, self.cube_quat))) < np.cos(0.12 / 2):
            raise RuntimeError("Cube tilted")
        if self.phase in ("hover", "descend") and np.linalg.norm(cube - self.cube_initial) > 0.002:
            raise RuntimeError("Cube moved before grasp")
        if self.carry_offset is not None:
            slip = float(np.linalg.norm(cube - self.grasp() - self.carry_offset))
            self.peak_slip = max(self.peak_slip, slip)
            if slip > 0.004 or not np.allclose(q[self.fi], [0.02508, 0.02508], atol=0.0015):
                raise RuntimeError(f"Grip lost or cube slipped: {slip}")
        if self.phase == "transfer" and cube[2] - 0.025 < 0.09:
            raise RuntimeError("Cube below required rim clearance during transfer")
        if self.released_cube is not None and np.linalg.norm(cube - self.released_cube) > 0.003:
            raise RuntimeError("Cube moved during retreat")
        if now - self.last_publish >= 0.1:
            message = JointState()
            message.header.stamp.sec = int(now)
            message.header.stamp.nanosec = int((now % 1) * 1e9)
            message.name = self.names
            message.position = q.tolist()
            message.velocity = self.arm.get_dof_velocities()[0].tolist()
            self.feedback.publish(message)
            self.last_publish = now

    def hold(self, duration):
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < duration:
            self.tick()

    def fk(self, angles):
        return link_transform(self.urdf, dict(zip(JOINT_NAMES, angles)))

    def ik(self, world_xyz, seed):
        target = np.asarray(world_xyz) - self.base
        indices = np.array([0, 1, 3, 4, 5, 6])
        desired = np.diag([1.0, -1.0, -1.0])

        def residual(values):
            angles = np.zeros(7)
            angles[indices] = values
            frame = self.fk(angles)
            error_r = Rotation.from_matrix(desired.T @ frame[:3, :3]).as_rotvec()
            return np.r_[frame[:3, 3] - target, 0.2 * error_r]

        bounds = np.asarray(LIMITS)[indices]
        result = least_squares(residual, np.clip(seed[indices], bounds[:, 0] + 1e-5, bounds[:, 1] - 1e-5),
                               bounds=(bounds[:, 0] + 1e-5, bounds[:, 1] - 1e-5), max_nfev=120,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not result.success or np.linalg.norm(result.fun) > 0.0001:
            raise RuntimeError(f"IK failed for {world_xyz}: {result.message}")
        angles = np.zeros(7)
        angles[indices] = result.x
        ordered_positions(JOINT_NAMES, angles)
        return angles

    def move(self, phase, target, minimum_duration=6.0):
        self.emit(phase, target_grasp_world_xyz=list(target))
        start_q = self.arm_target.copy()
        origin = self.fk(start_q)[:3, 3] + self.base
        samples = [start_q]
        for fraction in np.linspace(0, 1, 31)[1:]:
            samples.append(self.ik(origin + fraction * (np.asarray(target) - origin), samples[-1]))
        samples = np.asarray(samples)
        if np.max(np.abs(np.diff(samples, axis=0))) > 0.12:
            raise RuntimeError("IK branch discontinuity")
        duration = max(minimum_duration, float(np.linalg.norm(np.asarray(target) - origin)) / 0.025,
                       float(np.max(np.abs(np.diff(samples, axis=0)))) * 30 * 1.5 / 0.12)
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < duration:
            fraction = min((self.timeline.get_current_time() - start) / duration, 1)
            blend = fraction * fraction * (3 - 2 * fraction)
            coordinate = blend * 30
            index = min(int(coordinate), 29)
            self.set_arm(samples[index] + (coordinate - index) * (samples[index + 1] - samples[index]))
            self.tick()
        self.set_arm(samples[-1])
        self.hold(1.0)
        if np.linalg.norm(self.grasp() - target) > 0.008:
            raise RuntimeError(f"Grasp target error at {phase}: {self.grasp()}")

    def set_arm(self, angles):
        self.arm_target = np.asarray(angles).copy()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for drive, angle in zip(self.arm_drives, angles):
                drive.GetTargetPositionAttr().Set(float(np.degrees(angle)))

    def fingers_to(self, target):
        initial = self.q()[self.fi].copy()
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < 4:
            s = min((self.timeline.get_current_time() - start) / 4, 1)
            blend = s * s * (3 - 2 * s)
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                for drive, value in zip(self.finger_drives, initial + blend * (target - initial)):
                    drive.GetTargetPositionAttr().Set(float(value))
            self.tick()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for drive in self.finger_drives:
                drive.GetTargetPositionAttr().Set(float(target))
        self.hold(2.0)

    def bilateral(self):
        return all(any(finger in actors and "/World/Cube" in actors
                       for actors, colliders in self.active_contacts) for finger in self.fingers)

    def verify_inside(self):
        p, _ = self.pose("/World/Cube")
        # Entire axis-aligned 5 cm cube fits with margin; tilt separately limited.
        if np.any(np.abs(p[:2] - self.box) > 0.05) or abs(p[2] - 0.033) > 0.002:
            raise RuntimeError(f"Cube not resting inside box: {p}")
        supported = any("/World/Cube" in actors and any("/PlacementBox/Bottom" in c for c in colliders)
                        for actors, colliders in self.active_contacts)
        if not supported:
            raise RuntimeError("No confirmed cube/box-bottom contact")
        return p

    def run(self):
        self.move("hover", [*self.cube_initial[:2], 0.15], 12)
        self.move("descend", [*self.cube_initial[:2], 0.030], 10)
        self.emit("close")
        self.fingers_to(0.023)
        if not self.bilateral():
            raise RuntimeError("Bilateral grasp contact missing")
        for _ in range(30):
            self.tick()
            if not self.bilateral():
                raise RuntimeError("Grasp contact not sustained")
        cube, _ = self.pose("/World/Cube")
        if np.linalg.norm(cube - self.cube_initial) > 0.005:
            raise RuntimeError("Cube shifted too far during closure")
        self.carry_offset = cube - self.grasp()
        self.emit("grasp_verified", cube_world_xyz=cube.tolist())
        self.move("lift", [*self.cube_initial[:2], 0.18], 10)
        self.move("transfer", [*self.box, 0.18], 12)
        # Leave a few mm for release rather than forcing the cube into the floor.
        self.move("lower", [*self.box, 0.040], 10)
        cube, _ = self.pose("/World/Cube")
        if np.any(np.abs(cube[:2] - self.box) > 0.015) or not 0.031 <= cube[2] <= 0.048:
            raise RuntimeError("Release pose is not low and centred inside box")
        self.emit("release")
        self.carry_offset = None
        self.fingers_to(0.04)
        if not np.allclose(self.q()[self.fi], [0.04, 0.04], atol=0.001):
            raise RuntimeError("Fingers did not open")
        if any("/World/Cube" in a and any(f in a for f in self.fingers) for a, c in self.active_contacts):
            raise RuntimeError("Cube still contacting fingers after release")
        self.released_cube = self.verify_inside()
        self.emit("released", cube_world_xyz=self.released_cube.tolist())
        self.move("retreat", [*self.box, 0.18], 10)
        self.emit("verify")
        self.hold(3.0)
        final = self.verify_inside()
        report = {"completed": True, "cube_world_xyz": final.tolist(),
                  "cube_inside_box": True, "box_bottom_contact": True,
                  "fingers_open": True, "gripper_retreated": True,
                  "peak_joint_tracking_rad": self.peak_tracking,
                  "peak_carry_slip_metres": self.peak_slip, "events": self.events.copy(),
                  "scope": "Finite-step physics simulation only; no hardware or general planner guarantee."}
        self.emit("completed", cube_world_xyz=final.tolist())
        return report