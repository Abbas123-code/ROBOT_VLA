"""Bounded bottle lift driven by an RGB-D estimate; physics truth is guard-only.

Reuses the tested Panda initialization, IK, interpolation and pause service.
Does not alter the red-cube controller or its saved validation fingerprints.
"""
import numpy as np
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools

from pick_place_controller import PickPlaceNode

BOTTLE = "/World/TabletopProps/WaterBottle"


class BottlePickNode(PickPlaceNode):
    def __init__(self, app, urdf, detection):
        self.bottle_initial = None
        self.bottle_carry_offset = None
        self.bottle_peak_slip = 0.0
        self.obstacles = {}
        self.target_xy = np.asarray(detection["center_xy"], dtype=float)
        if self.target_xy.shape != (2,) or not np.isfinite(self.target_xy).all():
            raise RuntimeError("Invalid vision target")
        # This is the tested tabletop workspace envelope, not a target coordinate.
        if not (.35 < self.target_xy[0] < .8 and -.4 < self.target_xy[1] < .4):
            raise RuntimeError("Vision target outside tested tabletop workspace")
        super().__init__(app, urdf)
        self.bottle_initial, self.bottle_quat = self.pose(BOTTLE)
        self.vision_error = float(np.linalg.norm(self.bottle_initial[:2] - self.target_xy))
        # Ground truth rejects stale/mismatched snapshots; never corrects vision.
        if self.vision_error > .006 or abs(self.bottle_initial[2]) > .003:
            raise RuntimeError(f"Vision/physics safety check failed: {self.vision_error} m")
        for path in ("/World/Cube", *(f"/World/TabletopProps/{name}" for name in
                      ("GreenCube", "BlackCube", "OrangeCan", "YellowBlock"))):
            self.obstacles[path] = self.pose(path)
        self.emit("bottle_ready", image_target_xy=self.target_xy.tolist(),
                  vision_guard_error_m=self.vision_error)

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
                base_support = len(robots) == 1 and robots[0] == self.paths["panda_link0"] and any(
                    c.startswith("/World/table/") for c in (*actors, *colliders))
                finger_grip = len(robots) == 1 and robots[0] in self.fingers and BOTTLE in actors and self.phase in (
                    "bottle_close", "bottle_grasp_verified", "bottle_lift", "bottle_hold", "completed")
                if not (base_support or finger_grip):
                    self.unexpected.add(str((self.phase, actors, colliders)))
            if BOTTLE in actors:
                other = [a for a in actors if a != BOTTLE]
                if any(not (a in self.fingers or a.startswith("/World/table/")) for a in other):
                    self.unexpected.add("Bottle hit obstacle: " + str((actors, colliders)))

    def tick(self):
        super().tick()
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
            if slip > .004 or not self.bilateral():
                raise RuntimeError(f"Bottle grip lost/slipped: {slip} m")
            if np.any(self.q()[self.fi] < .026) or np.any(self.q()[self.fi] > .038):
                raise RuntimeError("Finger width inconsistent with bottle body")

    def bilateral(self):
        return all(any(finger in actors and BOTTLE in actors for actors, _ in self.active_contacts)
                   for finger in self.fingers)

    def run(self):
        # Body grasp: palm remains above the cap, fingers straddle the cylinder.
        # IK preflight: 23 cm exceeds reach for the measured target. 22 cm
        # is reachable and leaves 18 mm above the known 202 mm bottle cap.
        # Subdivide the far-reach approach instead of relaxing the existing
        # 0.12 rad branch-step limit near the arm's extension boundary.
        self.move("bottle_hover", [*(self.target_xy * .85), .22], 18)
        self.move("bottle_hover", [*(self.target_xy * .95), .22], 8)
        self.move("bottle_hover", [*self.target_xy, .22], 18)
        self.move("bottle_descend", [*self.target_xy, .115], 12)
        self.emit("bottle_close")
        self.fingers_to(.027)
        if not self.bilateral():
            raise RuntimeError("No two-sided bottle contact")
        self.hold(.5)
        p, _ = self.pose(BOTTLE)
        if np.linalg.norm(p - self.bottle_initial) > .008:
            raise RuntimeError("Bottle shifted excessively during finger closure")
        self.bottle_carry_offset = p - self.grasp()
        self.emit("bottle_grasp_verified", bottle_world_xyz=p.tolist())
        grasp_start = self.grasp().copy()
        self.move("bottle_lift", grasp_start + [0, 0, .05], 10)
        self.emit("bottle_hold")
        self.hold(3)
        final, _ = self.pose(BOTTLE)
        rise = float(final[2] - self.bottle_initial[2])
        if not .04 <= rise <= .06 or not self.bilateral():
            raise RuntimeError(f"Bottle lift not verified: rise={rise}")
        if any(BOTTLE in actors and any(c.startswith("/World/table/") for c in (*actors, *colliders))
               for actors, colliders in self.active_contacts):
            raise RuntimeError("Bottle still supported by table")
        report = {"completed": True, "vision_target_xy": self.target_xy.tolist(),
                  "vision_guard_error_m": self.vision_error, "bottle_final_xyz": final.tolist(),
                  "bottle_rise_m": rise, "bilateral_contact": True, "table_contact": False,
                  "peak_slip_m": self.bottle_peak_slip, "peak_tracking_rad": self.peak_tracking,
                  "other_objects_unchanged": True, "events": self.events.copy(),
                  "scope": "RGB-D snapshot target; simulator poses only for safety/verification; no visual servoing"}
        self.emit("completed", bottle_rise_m=rise)
        return report