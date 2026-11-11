"""Official YOLO RGB-D target -> angled body grasp -> upright box placement.

YOLO supplies targets, not motor commands. Legacy COCO uses the known box;
trained-tabletop mode requires detected tray XY/floor/rim (no fixed fallback).
Simulator poses supply carry-offset compensation as well as safety guards and
verification. This is snapshot targeting, not visual servoing or hardware control.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools, Usd, UsdGeom

from bottle_pick_controller import BOTTLE, BottlePickNode
from pick_place_controller import PickPlaceNode
from run_bottle_pick import TABLETOP_DETECTOR, validate_tabletop_detection
from src.control.bottle_grasp_plan import BottlePosePlanner, clearance_body_grasp
from src.control.contact_evidence import classify_contact
from vla_sim.motion import HOME, JOINT_NAMES, LIMITS


class YOLOBottlePlaceNode(BottlePickNode):
    def __init__(self, app, urdf, detection):
        self.trained_tabletop = detection.get("detector") == TABLETOP_DETECTOR
        if self.trained_tabletop:
            validate_tabletop_detection(detection)
            self.tray_floor_z = float(detection["tray_floor_z"])
            self.tray_rim_z = float(detection["tray_rim_z"])
            self.weights_sha256 = detection["weights_sha256"]
        else:
            if detection.get("detector") != "official-yolov9" or len(detection.get("frames", [])) != 5:
                raise RuntimeError("Five fresh official YOLO bottle observations required")
            for frame in detection["frames"]:
                item = frame.get("yolo_bottle", {})
                if frame.get("detector") != "WongKinYiu/yolov9" or item.get("label") != "bottle" or item.get("confidence", 0) < .5:
                    raise RuntimeError("Invalid YOLO bottle recognition evidence")
            self.tray_floor_z, self.tray_rim_z = .008, .06
        self.placed_bottle = None
        self.grip_diagnostics = []
        self.last_grip_sample = -1.
        self.loaded_finger_contacts = set()
        self.loaded_floor_contacts = set()
        self.proximity_events = 0
        super().__init__(app, urdf, detection)
        if self.trained_tabletop:
            # The superclass fixed-baseline fixture guard MUST run first.
            # This mode cannot relocate the physical fixture. Never replace a
            # rejected detection with simulator coordinates or widen that guard.
            fixture = self.stage.GetPrimAtPath("/World/PlacementBox")
            fixture_xy = np.asarray(UsdGeom.Xformable(fixture).ComputeLocalToWorldTransform(0).ExtractTranslation())[:2]
            detected_xy = np.asarray(detection["tray_center_xy"], dtype=float)
            self.tray_vision_error = float(np.linalg.norm(detected_xy - fixture_xy))
            if self.tray_vision_error > .008 or abs(self.tray_floor_z - .008) > .003:
                raise RuntimeError("Detected tray differs from baseline fixture by >8 mm XY or >3 mm floor")
            self.box = detected_xy.copy()
        # Preserve the original 90 mm minimum bottom height; a higher detected
        # rim can only strengthen clearance, never weaken the audited guard.
        self.transfer_min_bottom_z = max(.09, self.tray_rim_z + .03)
        self.transfer_grasp_z = max(.27, self.tray_rim_z + .21)
        self.planner = BottlePosePlanner(urdf, JOINT_NAMES, LIMITS)
        self.body_target, self.body_rotation = clearance_body_grasp(self.target_xy)
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
                self.loaded_finger_contacts.discard(key)
                self.loaded_floor_contacts.discard(key)
                continue
            if not header.num_contact_data:
                continue
            robots = [a for a in actors if a.startswith("/World/panda/")]
            offset = header.contact_data_offset
            separation = [float(data[i].separation) for i in range(offset, offset + header.num_contact_data)]
            impulses = [np.asarray(data[i].impulse, dtype=float).tolist()
                        for i in range(offset, offset + header.num_contact_data)]
            evidence = classify_contact(separation, impulses)
            if BOTTLE in actors and any(a.startswith("/World/PlacementBox/Bottom") for a in actors):
                if evidence == "loaded_touch":
                    self.loaded_floor_contacts.add(key)
                else:
                    self.loaded_floor_contacts.discard(key)
            placement_phases = ("bottle_lower", "bottle_release", "bottle_retreat", "bottle_verify", "return_home", "completed")
            table_pair = BOTTLE in actors and any(a.startswith("/World/table/") for a in actors)
            if table_pair and self.phase in placement_phases and evidence == "proximity":
                # Contact-offset overlap with the table under the 8 mm tray
                # floor is not table support. Any impulse/near-touch still aborts.
                self.active_contacts.discard(key)
                self.proximity_events += 1
                continue
            finger_bottle = len(robots) == 1 and robots[0] in self.fingers and BOTTLE in actors
            if finger_bottle:
                if evidence == "loaded_touch":
                    self.loaded_finger_contacts.add(key)
                else:
                    self.loaded_finger_contacts.discard(key)
                # Only the intended finger/body approach gets this distinction.
                # Palm, other objects and any nonzero speculative impulse retain
                # the original fail-closed policy. Proximity is NOT a grip.
                body_pair = any(c in (BOTTLE + "/Body", BOTTLE + "/Shoulder") for c in colliders)
                if evidence == "proximity" and body_pair and self.phase in ("bottle_descend", "bottle_release", "bottle_retreat", "bottle_verify", "return_home"):
                    self.active_contacts.discard(key)
                    self.proximity_events += 1
                    continue
            self.active_contacts.add(key)
            if robots:
                base = len(robots) == 1 and robots[0] == self.paths["panda_link0"] and any(
                    c.startswith("/World/table/") for c in (*actors, *colliders))
                grip = len(robots) == 1 and robots[0] in self.fingers and BOTTLE in actors and self.phase in (
                    "body_close", "body_verified", "body_lift", "bottle_raise",
                    "bottle_transfer", "bottle_lower", "bottle_release")
                if not (base or grip):
                    self.unexpected.add(str((self.phase, actors, colliders,
                                            {"separation_m": separation, "impulse_ns": impulses})))
            if BOTTLE in actors:
                for other in [a for a in actors if a != BOTTLE]:
                    if other in self.fingers:
                        continue
                    if other.startswith("/World/table/") and self.phase in (
                        "initializing", "ready", "bottle_ready", "bottle_hover", "bottle_descend", "body_close", "body_verified", "body_lift"):
                        continue
                    if other.startswith("/World/PlacementBox/Bottom") and self.phase in (
                        "bottle_lower", "bottle_release", "bottle_retreat", "bottle_verify", "return_home", "completed"):
                        continue
                    self.unexpected.add("Bottle obstacle contact: " + str((self.phase, actors, colliders,
                                        {"separation_m": separation, "impulse_ns": impulses})))

    def bilateral(self):
        # Require loaded near-touch evidence on BOTH fingers, not contact-offset
        # overlap. Persist through sleeping-body silence until LOST/new evidence.
        return all(any(finger in actors and BOTTLE in actors
                       for actors, _ in self.loaded_finger_contacts) for finger in self.fingers)

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
        if self.phase == "bottle_transfer" and p[2] < self.transfer_min_bottom_z:
            raise RuntimeError("Bottle bottom lacks box-rim clearance")
        if self.placed_bottle is not None and np.linalg.norm(p - self.placed_bottle) > .003:
            raise RuntimeError("Placed bottle moved during retreat")
        if self.phase == "return_home" and not np.allclose(self.q()[self.fi], [.04, .04], atol=.001):
            raise RuntimeError("Return home requires open fingers")
        camera = getattr(self, "live_camera", None)
        if camera is not None:
            camera.capture()

    def verify_placed(self):
        p, _ = self.pose(BOTTLE)
        if np.any(np.abs(p[:2] - self.box) > .04) or abs(p[2] - self.tray_floor_z) > .002:
            raise RuntimeError(f"Bottle is not supported inside box: {p}")
        floor = any(BOTTLE in actors and any("/PlacementBox/Bottom" in c for c in (*actors, *colliders))
                    for actors, colliders in self.loaded_floor_contacts)
        touching_fingers = any(BOTTLE in actors and any(f in actors for f in self.fingers)
                               for actors, _ in self.active_contacts)
        if not floor or touching_fingers:
            raise RuntimeError("Bottle support/release contact verification failed")
        return p

    def return_home(self, rotation, home_rotation):
        """Open-gripper, guarded retreat above the tray then exact HOME joints."""
        self.verify_placed()
        home = np.asarray(HOME, dtype=float)
        home_xyz = self.planner.fk(home)[:3, 3] + self.base
        for point, orientation in ((np.r_[self.box, .35], rotation),
                                   (np.array([.5, .23, .35]), rotation),
                                   (np.array([.5, .23, .35]), home_rotation),
                                   (home_xyz, home_rotation)):
            self.move_pose("return_home", point, orientation, 12)
        origin = self.arm_target.copy()
        # Redundant IK can reach HOME pose with different joints. Only a small
        # final branch correction is permitted; never jump or reset joints.
        if np.max(np.abs(home-origin)) > .15:
            raise RuntimeError("HOME branch correction exceeds planned 0.15 rad")
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < 8:
            s = min((self.timeline.get_current_time()-start)/8, 1.)
            self.set_arm(origin + (s*s*(3-2*s))*(home-origin))
            self.tick()
        self.set_arm(home)
        self.hold(2)
        if np.max(np.abs(self.q()[self.ai]-home)) > .01:
            raise RuntimeError("Measured HOME joints outside 0.01 rad")
        self.verify_placed()

    def run(self):
        rotation = self.body_rotation
        body = self.body_target.copy()
        pregrasp = body - .14 * rotation[:, 2]
        hover = np.array([.5, .23, .35])
        # Preflight the entire nominal route before moving any arm joint.
        # This proves reachability/continuity, not collision freedom; runtime
        # contact, tilt and neighbour-motion guards still decide that.
        seed = self.arm_target.copy()
        home_rotation = self.planner.fk(seed)[:3, :3]
        transfer_z = self.transfer_grasp_z
        nominal_lower_z = body[2] + self.tray_floor_z + .004
        # Include the known planned lateral offset in nominal carry preflight.
        # Actual lowering still uses the guarded measured carry offset.
        nominal_lower = np.r_[self.box + body[:2] - self.target_xy, nominal_lower_z]
        lift_target = body + [0, 0, .015]
        goals = [(hover, home_rotation), (hover, rotation), (pregrasp, rotation),
             (body, rotation), (lift_target, rotation),
             (np.r_[self.target_xy, transfer_z], rotation), (np.array([.5, .20, transfer_z]), rotation),
             (np.r_[self.box, transfer_z], rotation), (nominal_lower, rotation),
             (nominal_lower - .10 * rotation[:, 2], rotation)]
        goals += [(np.r_[self.box, .35], rotation), (hover, rotation),
                  (hover, home_rotation), (self.planner.fk(np.asarray(HOME))[:3, 3] + self.base, home_rotation)]
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
        measured_carry_offset = self.bottle_carry_offset.copy()
        self.emit("body_verified", bottle_xyz=p.tolist(), grasp_xyz=self.grasp().tolist())
        # Match the preflighted command-space lift. Using the measured sagged
        # grasp as a new command origin compounded static tracking error and
        # produced only 8.3 mm bottle rise from a nominal 15 mm lift.
        self.move_pose("body_lift", lift_target, rotation, 8)
        lifted, _ = self.pose(BOTTLE)
        if lifted[2] - self.bottle_initial[2] < .01:
            raise RuntimeError("Initial bottle lift failed")
        self.move_pose("bottle_raise", [*self.target_xy, transfer_z], rotation, 10)
        self.move_pose("bottle_transfer", [.5, .20, transfer_z], rotation, 10)
        self.move_pose("bottle_transfer", [*self.box, transfer_z], rotation, 20)
        # Detected floor (known floor in legacy mode) + unchanged 4 mm gap.
        # Simulator-measured XYZ carry offset actively compensates this target.
        target = np.r_[self.box, self.tray_floor_z + .004] - self.bottle_carry_offset
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
        self.return_home(rotation, home_rotation)
        final = self.verify_placed()
        report = {"completed": True, "vision_target_xy": self.target_xy.tolist(),
              "returned_home": True, "home_joint_error_rad": float(np.max(np.abs(self.q()[self.ai]-HOME))),
                  "vision_guard_error_m": self.vision_error, "bottle_final_xyz": final.tolist(),
                  "bottle_inside_box": True, "box_bottom_contact": True, "fingers_open": True,
                  "gripper_retreated": True, "other_objects_unchanged": True,
                  "peak_slip_m": self.bottle_peak_slip, "peak_tracking_rad": self.peak_tracking,
                  "finger_force_limit_n": 8.0,
                  "ignored_zero_impulse_body_proximity_events": self.proximity_events,
                  "grasp_strategy": "60-degree tilt, 20-degree world yaw, 5 mm inward radial offset, 143 mm height",
                  "planned_body_grasp_xyz": body.tolist(),
                  "grip_diagnostics": self.grip_diagnostics,
                  "events": self.events.copy(), "detector": "WongKinYiu/yolov9",
                  "scope": "YOLO + RGB-D snapshot targeting; known box; simulator carry-offset and guards; no visual servoing"}
        if self.trained_tabletop:
            report.update(detector=TABLETOP_DETECTOR, weights_sha256=self.weights_sha256,
                          tray_center_xy=self.box.tolist(), tray_floor_z=self.tray_floor_z, tray_rim_z=self.tray_rim_z,
                          detected_tray={"center_xy": self.box.tolist(), "floor_z": self.tray_floor_z,
                                         "rim_z": self.tray_rim_z},
                          tray_vision_guard_error_m=self.tray_vision_error,
                          simulator_carry_offset_xyz=measured_carry_offset.tolist(),
                          fixture_baseline_required=True,
                          scope="Trained YOLO + RGB-D bottle/tray snapshot targets; fixed baseline fixture still required; "
                                "simulator bottle/grasp poses actively supply XYZ carry-offset compensation for lowering "
                                "and also safety/contact/support verification; no visual servoing; simulation only")
        self.emit("completed", bottle_world_xyz=final.tolist())
        return report