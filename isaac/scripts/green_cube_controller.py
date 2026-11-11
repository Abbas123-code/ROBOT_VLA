"""Bounded reconstructed-scene green pick, loaded AFTER SimulationApp.

Only drive targets are written after playback. No teleport, attachment, retry or
failure release. The launcher must pause on exceptions and reconstruct the
bottle in the session layer BEFORE construction. No YOLO targeting is used.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools, UsdPhysics

from pick_place_controller import PickPlaceNode
from src.control.bottle_grasp_plan import BottlePosePlanner
from src.control.contact_evidence import classify_contact
from src.control.green_cube_plan import (
    BOTTLE, CENTRAL, DOWN, FLOOR, GREEN, START, TARGET, angle_between,
    bounds, nominal_goals, placement_geometry, preflight, vector,
)
from vla_sim.motion import HOME, JOINT_NAMES, LIMITS


class GreenCubeNode(PickPlaceNode):
    def __init__(self, app, urdf, bottle_xyz):
        self.expected_bottle = vector(bottle_xyz)
        placement_geometry(self.expected_bottle)
        # Contact callbacks run during the parent's warmup, before live poses.
        self.loaded_finger_contacts = set()
        self.loaded_floor_contacts = set()
        self.green_carry_offset = None
        self.placed_green = None
        self.green_peak_slip = 0.
        self.proximity_events = 0
        self.green_start = None
        super().__init__(app, urdf)
        # Do not change cube_initial/cube_quat or the parent's 5 N drives.
        # Capture ALL non-target rigid props AFTER its 120-step warmup.
        paths = {"/World/Cube", BOTTLE}
        paths.update(str(p.GetPath()) for p in self.stage.Traverse()
                     if str(p.GetPath()).startswith("/World/TabletopProps/")
                     and p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()) != GREEN)
        self.obstacles = {path: self.pose(path) for path in paths}
        bottle, bq = self.obstacles[BOTTLE]
        if np.linalg.norm(bottle - self.expected_bottle) > .002 or abs(bottle[2] - FLOOR) > .002:
            raise RuntimeError("Reconstructed bottle moved >2 mm during warmup")
        self.green_start, self.green_rotation = self.pose(GREEN)
        if (np.linalg.norm(self.green_start[:2] - START[:2]) > .002
                or abs(self.green_start[2]) > .002
            or abs(bounds(self.green_start, self.green_rotation)[0][2]) > .002
                or angle_between(self.green_rotation, (1., 0., 0., 0.)) > .10):
            raise RuntimeError("Green root is not at baseline XY/bottom Z or is tilted")
        self.geometry = placement_geometry(bottle, bq, TARGET, self.green_rotation)
        self.planner = BottlePosePlanner(urdf, JOINT_NAMES, LIMITS)
        self.emit("green_ready", targeting="simulator-known reconstructed fixture; NOT YOLO")
        self.tick()

    def contacts(self, headers, data):
        placement = ("green_lower", "green_release", "green_retreat", "green_home", "green_verify", "green_completed")
        initial = ("initializing", "ready", "green_ready", "green_hover", "green_descend", "green_close", "green_verified", "green_lift")
        grip_phases = ("green_close", "green_verified", "green_lift", "green_transfer", "green_lower", "green_release")
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
                continue  # Sleeping silence does not erase loaded support.
            points = [data[i] for i in range(header.contact_data_offset,
                                             header.contact_data_offset + header.num_contact_data)]
            gaps = np.asarray([float(p.separation) for p in points])
            impulses = np.asarray([np.asarray(p.impulse, dtype=float) for p in points])
            valid = (impulses.shape == (len(points), 3) and np.isfinite(gaps).all()
                     and np.isfinite(impulses).all())
            evidence = classify_contact(gaps, impulses)
            robots = [a for a in actors if a.startswith("/World/panda/")]
            table = any(p.startswith("/World/table/") for p in (*actors, *colliders))
            floor = any(p.startswith("/World/PlacementBox/Bottom") for p in (*actors, *colliders))
            rim = any(p.startswith("/World/PlacementBox/") and not p.startswith("/World/PlacementBox/Bottom")
                      for p in (*actors, *colliders))
            finger_green = len(robots) == 1 and robots[0] in self.fingers and GREEN in actors
            for matches, collection in ((finger_green, self.loaded_finger_contacts),
                                        (GREEN in actors and floor and not rim, self.loaded_floor_contacts)):
                if matches and evidence == "loaded_touch":
                    collection.add(key)
                else:
                    collection.discard(key)
            permitted = False
            proximity = False
            if finger_green:
                proximity = evidence == "proximity" and self.phase in (
                    "green_descend", "green_close", "green_release", "green_retreat", "green_home")
                permitted = proximity or (evidence != "proximity" and self.phase in grip_phases)
            elif robots:
                permitted = len(robots) == 1 and robots[0] == self.paths["panda_link0"] and table
            elif GREEN in actors:
                # Occupied-box descent can overlap contact offsets while the
                # solids remain separated. Never count this as tray support.
                # tick() independently checks actual rotated footprint clearance
                # and bottle motion on every simulation step.
                bottle_pair = set(actors) == {GREEN, BOTTLE}
                proximity = evidence == "proximity" and (
                    (table and self.phase in (*placement, "green_lift"))
                    or (bottle_pair and self.phase in placement))
                permitted = (proximity or (table and self.phase in initial and evidence != "proximity")
                             or (floor and not rim and self.phase in placement))
            else:
                # Only established fixture support is benign; other object,
                # neighbour, wall and rim pairs fail closed, even in warmup.
                props = [a for a in actors if a == "/World/Cube" or a.startswith("/World/TabletopProps/")]
                if len(props) == 1:
                    permitted = (table and props[0] != BOTTLE) or (
                        props[0] == BOTTLE and ((floor and not rim) or (table and evidence == "proximity")))
            if proximity:
                self.active_contacts.discard(key)
                self.proximity_events += 1
            else:
                self.active_contacts.add(key)
            if not permitted or not valid:
                self.unexpected.add(str((self.phase, actors, colliders, evidence,
                                        {"separation_m": gaps.tolist(), "impulse_ns": impulses.tolist()})))

    def bilateral(self):
        return all(any(f in actors and GREEN in actors for actors, _ in self.loaded_finger_contacts)
                   for f in self.fingers)

    def tick(self):
        if self.carry_offset is not None or self.released_cube is not None:
            raise RuntimeError("Parent red-cube carry state must remain unused")
        PickPlaceNode.tick(self)
        if self.green_start is None:
            return
        p, quat = self.pose(GREEN)
        if (angle_between(quat, self.green_rotation) > .10
                or angle_between(quat, (1., 0., 0., 0.)) > .10):
            raise RuntimeError("Green tilt exceeds .10 rad")
        if self.phase in ("green_ready", "green_hover", "green_descend") and np.linalg.norm(p - self.green_start) > .002:
            raise RuntimeError("Green moved before grasp")
        for path, (start, rotation) in self.obstacles.items():
            now, q = self.pose(path)
            if np.linalg.norm(now - start) > .003 or angle_between(q, rotation) > .05:
                raise RuntimeError("Protected non-target prop moved: " + path)
        bottle, bq = self.pose(BOTTLE)
        # Test projected occupancy while carrying AND actual occupancy in tray.
        self.geometry = placement_geometry(bottle, bq, TARGET, quat)
        if self.phase in ("green_lower", "green_release", "green_retreat", "green_home", "green_verify", "green_completed"):
            self.geometry = placement_geometry(bottle, bq, p, quat)
        if self.green_carry_offset is not None:
            slip = float(np.linalg.norm(p - self.grasp() - self.green_carry_offset))
            self.green_peak_slip = max(self.green_peak_slip, slip)
            if (slip > .004 or not self.bilateral()
                    or not np.allclose(self.q()[self.fi], [.02508, .02508], atol=.0015, rtol=0)):
                raise RuntimeError(f"Green grip lost/width invalid/slip >4 mm: {slip}")
        if self.phase == "green_transfer" and bounds(p, quat)[0][2] < .09:
            raise RuntimeError("Green lacks 90 mm bottom transfer clearance")
        if self.placed_green is not None:
            if np.linalg.norm(p - self.placed_green) > .003:
                raise RuntimeError("Placed green moved during retreat")
            self.verify_placed()
            if not np.allclose(self.q()[self.fi], [.04, .04], atol=.001, rtol=0):
                raise RuntimeError("Retreat/HOME requires open fingers")

    def move_pose(self, phase, target, orientation, minimum_duration=10.):
        target = vector(target)
        samples = self.planner.segment(self.arm_target, target - self.base, orientation)
        self.emit(phase, target_grasp_world_xyz=target.tolist())
        count = len(samples) - 1
        origin = self.planner.fk(samples[0])[:3, 3] + self.base
        duration = max(minimum_duration, float(np.linalg.norm(target - origin)) / .025,
                       float(np.max(np.abs(np.diff(samples, axis=0)))) * count * 1.5 / .12)
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < duration:
            s = min((self.timeline.get_current_time() - start) / duration, 1.)
            coordinate = s * s * (3 - 2 * s) * count
            i = min(int(coordinate), count - 1)
            self.set_arm(samples[i] + (coordinate - i) * (samples[i + 1] - samples[i]))
            self.tick()
        self.set_arm(samples[-1])
        self.hold(1.)
        measured = self.planner.fk(self.q()[self.ai])
        error_r = Rotation.from_matrix(orientation.T @ measured[:3, :3]).magnitude()
        if np.linalg.norm(self.grasp() - target) > .008 or error_r > .04:
            raise RuntimeError(f"Green pose tracking failed: {phase}")

    def preflight(self):
        """No drive commands; called by run() before any task motion."""
        if np.max(np.abs(self.arm_target - np.asarray(HOME))) > 1e-8:
            raise RuntimeError("Nominal preflight requires HOME command seed")
        self.preflight_report = preflight(self.planner, HOME, self.expected_bottle, self.base)
        return self.preflight_report

    def verify_placed(self):
        p, quat = self.pose(GREEN)
        bottle, bq = self.pose(BOTTLE)
        geometry = placement_geometry(bottle, bq, p, quat)
        floor = any(GREEN in actors and any(c.startswith("/World/PlacementBox/Bottom") for c in (*actors, *colliders))
                    for actors, colliders in self.loaded_floor_contacts)
        touching = any(GREEN in actors and any(f in actors for f in self.fingers)
                       for actors, _ in self.active_contacts)
        if (np.linalg.norm(p[:2] - TARGET[:2]) > .003 or abs(p[2] - FLOOR) > .002
                or abs(geometry["green_bottom_z"] - FLOOR) > .002 or not floor or touching):
            raise RuntimeError("Green placement lacks loaded floor support, clearance or finger release")
        return p.copy()

    def return_home(self):
        home = np.asarray(HOME)
        frame = self.planner.fk(home)
        for point, orientation in ((CENTRAL, DOWN),
                                   (CENTRAL, frame[:3, :3]), (frame[:3, 3] + self.base, frame[:3, :3])):
            self.move_pose("green_home", point, orientation, 12.)
        origin = self.arm_target.copy()
        if np.max(np.abs(home - origin)) > .15:
            raise RuntimeError("HOME branch correction exceeds .15 rad")
        start = self.timeline.get_current_time()
        while self.timeline.get_current_time() - start < 8.:
            s = min((self.timeline.get_current_time() - start) / 8., 1.)
            self.set_arm(origin + s * s * (3 - 2 * s) * (home - origin))
            self.tick()
        self.set_arm(home)
        self.hold(2.)
        if np.max(np.abs(self.q()[self.ai] - home)) > .01:
            raise RuntimeError("Measured HOME error exceeds .01 rad")
        self.verify_placed()

    def run(self):
        self.preflight()  # Full route including HOME must pass before motion.
        goals = nominal_goals(self.planner, HOME, self.base)
        for phase, point, orientation in goals[:4]:
            self.move_pose(phase, point, orientation)
        self.emit("green_close")
        self.fingers_to(.023)
        for _ in range(30):
            self.tick()
            if not self.bilateral():
                raise RuntimeError("Bilateral loaded green grip missing")
        p, _ = self.pose(GREEN)
        if np.linalg.norm(p - self.green_start) > .005:
            raise RuntimeError("Green shifted >5 mm during closure")
        self.green_carry_offset = p - self.grasp()
        offset = self.green_carry_offset.copy()
        self.emit("green_verified")
        self.tick()
        for phase, point, orientation in goals[4:7]:
            self.move_pose(phase, point, orientation)
        lower = TARGET + [0., 0., .002] - offset
        self.move_pose("green_lower", lower, DOWN, 12.)
        p, quat = self.pose(GREEN)
        placement_geometry(*self.pose(BOTTLE), p, quat)
        if np.linalg.norm(p[:2] - TARGET[:2]) > .003 or not FLOOR <= p[2] <= FLOOR + .004:
            raise RuntimeError("Green is not at guarded release pose")
        self.emit("green_release")
        self.green_carry_offset = None
        self.fingers_to(.04)
        if not np.allclose(self.q()[self.fi], [.04, .04], atol=.001, rtol=0):
            raise RuntimeError("Green fingers did not open")
        self.placed_green = self.verify_placed()
        self.move_pose("green_retreat", np.r_[lower[:2], .32], DOWN)
        self.return_home()
        self.emit("green_verify")
        self.hold(3.)
        final = self.verify_placed()
        bottle, _ = self.pose(BOTTLE)
        error = float(np.max(np.abs(self.q()[self.ai] - np.asarray(HOME))))
        if error > .01 or self.grasp()[2] - (final[2] + .05) < .10:
            raise RuntimeError("Final HOME/retreat verification failed")
        report = {"completed": True, "green_xyz": final.tolist(), "returned_home": True,
                  "home_joint_error_rad": error, "bottle_xyz": bottle.tolist(), "bottle_preserved": True,
                  "box_bottom_contact": True, "fingers_open": True, "gripper_retreated": True,
                  "peak_slip_m": self.green_peak_slip, "peak_tracking_rad": self.peak_tracking,
                  "other_objects_unchanged": True, "finger_force_limit_n": 5.,
                  "geometry": self.geometry, "preflight": self.preflight_report,
                  "simulator_carry_offset_xyz": offset.tolist(),
                  "scope": "Reconstructed simulation; simulator-known fixture coordinates, NOT YOLO; "
                           "physics grip and measured carry offset; no hardware or collision-free guarantee"}
        self.emit("green_completed", completed=True, green_xyz=final.tolist())
        report["events"] = self.events.copy()
        return report