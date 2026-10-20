"""Official YOLO RGB-D target -> angled body grasp -> upright box placement.

YOLO supplies the bottle target, not motor commands. Box coordinates are the
known task fixture. Physics ground truth is used only for guards/verification.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools, Usd

from bottle_pick_controller import BOTTLE, BottlePickNode
from pick_place_controller import PickPlaceNode
from src.control.bottle_grasp_plan import BottlePosePlanner, body_orientation
from vla_sim.motion import JOINT_NAMES, LIMITS


class YOLOBottlePlaceNode(BottlePickNode):
    def __init__(self, app, urdf, detection):
        if detection.get("detector") != "official-yolov9" or len(detection.get("frames", [])) != 5:
            raise RuntimeError("Five fresh official YOLO bottle observations required")
        for frame in detection["frames"]:
            item = frame.get("yolo_bottle", {})
            if frame.get("detector") != "WongKinYiu/yolov9" or item.get("label") != "bottle" or item.get("confidence", 0) < .5:
                raise RuntimeError("Invalid YOLO bottle recognition evidence")
        self.placed_bottle = None
        self.grip_diagnostics = []
        self.last_grip_sample = -1.
        super().__init__(app, urdf, detection)
        self.planner = BottlePosePlanner(urdf, JOINT_NAMES, LIMITS)
        self.body_rotation = body_orientation(self.target_xy, tilt_degrees=45, roll_sign=-1)
        # Keep the already-audited force and material configuration. Change the
        # grasp geometry, not bottle physics or the 4 mm slip limit.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for drive in self.finger_drives:
                drive.CreateMaxForceAttr(8.0)

    def move_pose(self, phase, target, orientation, minimum_duration=6.):
        samples = self.planner.segment(self.arm_target, np.asarray(target) - self.base, orientation)
        self.emit(phase, target_grasp_world_xyz=np.asarray(target).tolist())
        origin = self.planner.fk(samples[0])[:3, 3] + self.base
        count = len(samples) - 1
        duration = max(minimum_duration, float(np.linalg.norm(np.asarray(target) - origin)) / .025,
                       float(np.max(np.abs(np.diff(samples, axis=0)))) * count * 1.5 / .12)
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < duration:
            s = min((self.timeline.get_current_time() - start) / duration, 1.)
            coordinate = s * s * (3 - 2 * s) * count
            index = min(int(coordinate), count - 1)
            self.set_arm(samples[index] + (coordinate - index) * (samples[index + 1] - samples[index]))
            self.tick()
        self.set_arm(samples[-1])
        self.hold(1.)
        measured = self.planner.fk(self.q()[self.ai])
        angle_error = np.linalg.norm(Rotation.from_matrix(orientation.T @ measured[:3, :3]).as_rotvec())
        if np.linalg.norm(self.grasp() - target) > .008 or angle_error > .04:
            raise RuntimeError(f"Body grasp pose tracking failed at {phase}: {self.grasp()}, rotation={angle_error}")

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
                base = len(robots) == 1 and robots[0] == self.paths["panda_link0"] and any(
                    c.startswith("/World/table/") for c in (*actors, *colliders))
                grip = len(robots) == 1 and robots[0] in self.fingers and BOTTLE in actors and self.phase in (
                    "body_close", "body_verified", "body_lift", "bottle_raise",
                    "bottle_transfer", "bottle_lower", "bottle_release")
                if not (base or grip):
                    offset = header.contact_data_offset
                    separation = [float(data[i].separation) for i in range(offset, offset + header.num_contact_data)]
                    self.unexpected.add(str((self.phase, actors, colliders, {"separation_m": separation})))
            if BOTTLE in actors:
                for other in [a for a in actors if a != BOTTLE]:
                    if other in self.fingers:
                        continue
                    if other.startswith("/World/table/") and self.phase in (
                        "initializing", "ready", "bottle_ready", "bottle_hover", "bottle_descend", "body_close", "body_verified", "body_lift"):
                        continue
                    if other.startswith("/World/PlacementBox/Bottom") and self.phase in (
                        "bottle_lower", "bottle_release", "bottle_retreat", "bottle_verify", "completed"):
                        continue
                    self.unexpected.add("Bottle obstacle contact: " + str((self.phase, actors, colliders)))

    def tick(self):
        # Keep the proven robot, cube, time, tracking, pause and contact guards.
        PickPlaceNode.tick(self)
        if self.bottle_initial is None:
            return
        p, q = self.pose(BOTTLE)
        if abs(float(np.dot(q, self.bottle_quat))) < np.cos(.10 / 2):
            raise RuntimeError("Bottle tilted by more than 0.10 rad")
        if self.phase in ("bottle_hover", "bottle_descend") and np.linalg.norm(p - self.bottle_initial) > .002:
            raise RuntimeError("Bottle moved before permitted grasp contact")
        for path, (start, rotation) in self.obstacles.items():
            now, quat = self.pose(path)
            if np.linalg.norm(now - start) > .003 or abs(float(np.dot(quat, rotation))) < np.cos(.05 / 2):
                raise RuntimeError("Neighbouring object moved: " + path)
        if self.bottle_carry_offset is not None:
            slip = float(np.linalg.norm(p - self.grasp() - self.bottle_carry_offset))
            self.bottle_peak_slip = max(self.bottle_peak_slip, slip)
            if self.last_time - self.last_grip_sample >= .1 or slip > .004:
                self.last_grip_sample = self.last_time
                self.grip_diagnostics.append({"time": self.last_time, "phase": self.phase,
                    "bottle_xyz": p.tolist(), "grasp_xyz": self.grasp().tolist(), "slip_m": slip,
                    "bilateral": self.bilateral(), "fingers_m": self.q()[self.fi].tolist(),
                    "solver_max_force_n": self.arm.get_dof_max_forces()[0, self.fi].tolist()})
            if slip > .004 or not self.bilateral():
                raise RuntimeError(f"Bottle grip lost/slipped: {slip} m")
            if np.any(self.q()[self.fi] < .026) or np.any(self.q()[self.fi] > .038):
                raise RuntimeError("Finger width inconsistent with bottle body")
        if self.phase == "bottle_transfer" and p[2] < .09:
            raise RuntimeError("Bottle bottom lacks box-rim clearance")
        if self.placed_bottle is not None and np.linalg.norm(p - self.placed_bottle) > .003:
            raise RuntimeError("Placed bottle moved during retreat")

    def verify_placed(self):
        p, _ = self.pose(BOTTLE)
        if np.any(np.abs(p[:2] - self.box) > .04) or abs(p[2] - .008) > .002:
            raise RuntimeError(f"Bottle is not supported inside box: {p}")
        floor = any(BOTTLE in actors and any("/PlacementBox/Bottom" in c for c in (*actors, *colliders))
                    for actors, colliders in self.active_contacts)
        touching_fingers = any(BOTTLE in actors and any(f in actors for f in self.fingers)
                               for actors, _ in self.active_contacts)
        if not floor or touching_fingers:
            raise RuntimeError("Bottle support/release contact verification failed")
        return p

    def run(self):
        rotation = self.body_rotation
        body = np.r_[self.target_xy, .13]
        pregrasp = body - .14 * rotation[:, 2]
        hover = np.array([.5, .23, .35])
        # Preflight the entire nominal route before moving any arm joint.
        # This proves reachability/continuity, not collision freedom; runtime
        # contact, tilt and neighbour-motion guards still decide that.
        seed = self.arm_target.copy()
        home_rotation = self.planner.fk(seed)[:3, :3]
        goals = [(hover, home_rotation), (hover, rotation), (pregrasp, rotation),
                 (body, rotation), (body + [0, 0, .015], rotation),
                 (np.r_[self.target_xy, .27], rotation), (np.array([.5, .20, .27]), rotation),
                 (np.r_[self.box, .27], rotation), (np.r_[self.box, .142], rotation),
                 (np.r_[self.box, .142] - .10 * rotation[:, 2], rotation)]
        for position, orientation in goals:
            seed = self.planner.segment(seed, position - self.base, orientation)[-1]
        self.move_pose("bottle_hover", hover, home_rotation, 14)
        self.move_pose("bottle_hover", hover, rotation, 14)
        self.move_pose("bottle_hover", pregrasp, rotation, 12)
        self.move_pose("bottle_descend", body, rotation, 10)
        self.emit("body_close")
        self.fingers_to(.024)
        if not self.bilateral():
            raise RuntimeError("No bilateral body grasp")
        self.hold(.5)
        p, _ = self.pose(BOTTLE)
        if np.linalg.norm(p - self.bottle_initial) > .008:
            raise RuntimeError("Bottle shifted excessively during closure")
        self.bottle_carry_offset = p - self.grasp()
        self.emit("body_verified", bottle_xyz=p.tolist(), grasp_xyz=self.grasp().tolist())
        start = self.grasp().copy()
        self.move_pose("body_lift", start + [0, 0, .015], rotation, 8)
        lifted, _ = self.pose(BOTTLE)
        if lifted[2] - self.bottle_initial[2] < .01:
            raise RuntimeError("Initial bottle lift failed")
        self.move_pose("bottle_raise", [*self.target_xy, .27], rotation, 10)
        self.move_pose("bottle_transfer", [.5, .20, .27], rotation, 10)
        self.move_pose("bottle_transfer", [*self.box, .27], rotation, 20)
        # Desired bottom 12mm = 8mm box floor + 4mm release gap.
        target = np.r_[self.box, .012] - self.bottle_carry_offset
        self.move_pose("bottle_lower", target, rotation, 12)
        self.emit("bottle_release")
        self.bottle_carry_offset = None
        self.fingers_to(.04)
        if not np.allclose(self.q()[self.fi], [.04, .04], atol=.001):
            raise RuntimeError("Gripper did not open")
        self.placed_bottle = self.verify_placed()
        self.move_pose("bottle_retreat", target - .10 * rotation[:, 2], rotation, 10)
        self.emit("bottle_verify")
        self.hold(3)
        final = self.verify_placed()
        report = {"completed": True, "vision_target_xy": self.target_xy.tolist(),
                  "vision_guard_error_m": self.vision_error, "bottle_final_xyz": final.tolist(),
                  "bottle_inside_box": True, "box_bottom_contact": True, "fingers_open": True,
                  "gripper_retreated": True, "other_objects_unchanged": True,
                  "peak_slip_m": self.bottle_peak_slip, "peak_tracking_rad": self.peak_tracking,
                  "finger_force_limit_n": 8.0,
                  "grasp_strategy": "45-degree side approach to upper cylindrical body",
                  "grip_diagnostics": self.grip_diagnostics,
                  "events": self.events.copy(), "detector": "WongKinYiu/yolov9",
                  "scope": "YOLO + RGB-D snapshot targeting; known box; simulator carry-offset and guards; no visual servoing"}
        self.emit("completed", bottle_world_xyz=final.tolist())
        return report