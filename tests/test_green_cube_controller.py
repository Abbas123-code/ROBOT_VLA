"""Run actual AST-extracted controller methods without Isaac or robot motion."""
import ast
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.contact_evidence import classify_contact
from src.control.green_cube_plan import (
    BOTTLE, FLOOR, GREEN, START, TARGET, angle_between, bounds, placement_geometry,
)

LEFT, RIGHT = "/World/panda/left", "/World/panda/right"
PALM = "/World/panda/hand"
TABLE = "/World/table/Geometry/baseLink/box"
BOTTOM = "/World/PlacementBox/Bottom"
BOTTLE_XYZ = np.array([.4466993213, -.2437739223, .0079996884])
ContactEventType = SimpleNamespace(CONTACT_FOUND=1, CONTACT_PERSIST=2, CONTACT_LOST=3)
PhysicsSchemaTools = SimpleNamespace(intToSdfPath=lambda x: x)
# Only the inherited framework tick is stubbed for own green safety tests;
# contacts(), bilateral(), verify_placed() and tick() themselves are real code.
def parent_tick(obj):
    obj.parent_ticks += 1
    if obj.parent_fault:
        raise RuntimeError("Inherited tracking/time/contact guard")


PickPlaceNode = SimpleNamespace(tick=parent_tick)
SOURCE = ROOT / "isaac/scripts/green_cube_controller.py"
TREE = ast.parse(SOURCE.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "GreenCubeNode")
METHODS = [n for n in CLASS.body if isinstance(n, ast.FunctionDef)
           and n.name in ("contacts", "bilateral", "verify_placed", "tick")]
exec(compile(ast.Module(body=METHODS, type_ignores=[]), str(SOURCE), "exec"), globals())


def node(phase="green_descend"):
    obj = SimpleNamespace(stage_id=1, phase=phase, fingers={LEFT, RIGHT},
                          paths={"panda_link0": "/World/panda/base"}, active_contacts=set(),
                          loaded_finger_contacts=set(), loaded_floor_contacts=set(),
                          unexpected=set(), proximity_events=0)
    for method in METHODS:
        setattr(obj, method.name, MethodType(globals()[method.name], obj))
    obj.pose = lambda path: ((TARGET if path == GREEN else BOTTLE_XYZ).copy(), np.array([1., 0., 0., 0.]))
    return obj


def report(obj, a=LEFT, b=GREEN, gap=.002, impulse=(0., 0., 0.),
           event=ContactEventType.CONTACT_FOUND, stage_id=1, reverse=False, empty=False,
           collider=None):
    actors = (a, b)
    colliders = (a + "/Collision", collider or b + "/Shape")
    if reverse:
        actors, colliders = actors[::-1], colliders[::-1]
    count = int(event != ContactEventType.CONTACT_LOST and not empty)
    header = SimpleNamespace(stage_id=stage_id, actor0=actors[0], actor1=actors[1],
                             collider0=colliders[0], collider1=colliders[1], type=event,
                             contact_data_offset=1, num_contact_data=count)
    # Nonzero data offset ensures unrelated contact points are not consumed.
    obj.contacts([header], [SimpleNamespace(separation=np.nan, impulse=(np.nan, 0, 0)),
                           SimpleNamespace(separation=gap, impulse=impulse)])
    return tuple(sorted(actors)), tuple(sorted(colliders))


class GreenContactsTests(unittest.TestCase):
    def test_only_intended_finger_proximity_phases(self):
        allowed = ("green_descend", "green_close", "green_release", "green_retreat", "green_home")
        denied = ("green_hover", "green_lift", "green_transfer", "green_lower", "green_verify")
        for phase in allowed + denied:
            for reverse in (False, True):
                obj = node(phase)
                report(obj, reverse=reverse)
                self.assertEqual(bool(obj.unexpected), phase in denied, phase)
                self.assertFalse(obj.bilateral())
                self.assertEqual(obj.proximity_events, int(phase in allowed))

    def test_two_mm_and_exact_zero_impulse_required(self):
        for gap, impulse in ((.001999, (0, 0, 0)), (.002, (1e-20, 0, 0)),
                             (0, (0, 0, 0)), (0, (0, .01, 0)), (np.nan, (0, 0, 0))):
            obj = node()
            report(obj, gap=gap, impulse=impulse)
            self.assertTrue(obj.unexpected)
            self.assertEqual(obj.proximity_events, 0)

    def test_bilateral_loaded_not_proximity_or_speculative(self):
        obj = node("green_close")
        report(obj, gap=0, impulse=(0, .01, 0))
        self.assertFalse(obj.bilateral())
        report(obj, a=RIGHT, gap=0)
        self.assertFalse(obj.bilateral())
        report(obj, a=RIGHT, gap=.001, impulse=(0, .01, 0))
        self.assertFalse(obj.bilateral())
        report(obj, a=RIGHT, gap=.0005, impulse=(0, .01, 0))
        self.assertTrue(obj.bilateral())
        self.assertFalse(obj.unexpected)
        obj.contacts([], [])
        report(obj, empty=True)
        self.assertTrue(obj.bilateral())
        report(obj, event=ContactEventType.CONTACT_LOST, reverse=True)
        self.assertFalse(obj.bilateral())

    def test_new_nonloaded_evidence_clears_grip(self):
        obj = node("green_close")
        for finger in (LEFT, RIGHT):
            report(obj, a=finger, gap=0, impulse=(0, .01, 0))
        self.assertTrue(obj.bilateral())
        obj.phase = "green_release"
        report(obj, event=ContactEventType.CONTACT_PERSIST)
        self.assertFalse(obj.bilateral())
        self.assertFalse(obj.unexpected)

    def test_floor_support_persists_until_lost_or_nonloaded(self):
        for transition in ("lost", "unloaded", "proximity"):
            obj = node("green_release")
            report(obj, a=BOTTOM, gap=0, impulse=(0, 0, .01))
            self.assertEqual(len(obj.loaded_floor_contacts), 1)
            obj.contacts([], [])
            report(obj, a=BOTTOM, empty=True)
            np.testing.assert_allclose(obj.verify_placed(), TARGET)
            if transition == "lost":
                report(obj, a=BOTTOM, event=ContactEventType.CONTACT_LOST)
            else:
                report(obj, a=BOTTOM, gap=0 if transition == "unloaded" else .002)
            self.assertFalse(obj.loaded_floor_contacts)
            with self.assertRaises(RuntimeError):
                obj.verify_placed()

    def test_bottle_support_is_not_green_support(self):
        obj = node("green_release")
        report(obj, a=BOTTOM, b=BOTTLE, gap=0, impulse=(0, 0, .01))
        self.assertFalse(obj.unexpected)
        self.assertFalse(obj.loaded_floor_contacts)
        with self.assertRaises(RuntimeError):
            obj.verify_placed()

    def test_table_proximity_during_placement_not_support(self):
        obj = node("green_lower")
        report(obj, a=TABLE, gap=.008)
        self.assertFalse(obj.unexpected)
        self.assertFalse(obj.active_contacts)
        self.assertFalse(obj.loaded_floor_contacts)
        for gap, impulse in ((.001, (0, 0, 0)), (.008, (0, 0, 1e-20)), (0, (0, 0, .1))):
            obj = node("green_lower")
            report(obj, a=TABLE, gap=gap, impulse=impulse)
            self.assertTrue(obj.unexpected)
        obj = node("green_transfer")
        report(obj, a=TABLE)
        self.assertTrue(obj.unexpected)
        obj = node("initializing")
        report(obj, a=TABLE)
        self.assertTrue(obj.unexpected)

    def test_recorded_liftoff_table_proximity_is_not_support(self):
        obj = node("green_lift")
        report(obj, a=TABLE, gap=.0020551905035972595)
        self.assertFalse(obj.unexpected)
        self.assertFalse(obj.active_contacts)
        self.assertFalse(obj.loaded_floor_contacts)
        self.assertFalse(obj.bilateral())
        obj.phase = "green_transfer"
        report(obj, a=TABLE, gap=.0020551905035972595)
        self.assertTrue(obj.unexpected)

    def test_nonfinite_contact_data_fails_even_for_expected_pairs(self):
        for a, phase in ((LEFT, "green_close"), (BOTTOM, "green_release"), (TABLE, "green_lift")):
            for gap, impulse in ((np.nan, (0, 0, 0)), (0, (0, np.inf, 0))):
                obj = node(phase)
                report(obj, a=a, gap=gap, impulse=impulse)
                self.assertTrue(obj.unexpected)
                self.assertFalse(obj.loaded_floor_contacts)
                self.assertFalse(obj.loaded_finger_contacts)

    def test_neighbours_palm_rim_robot_and_early_floor_are_forbidden(self):
        for a, b, phase in ((PALM, GREEN, "green_close"), (LEFT, BOTTLE, "green_release"),
                            (LEFT, RIGHT, "green_descend"), (GREEN, BOTTLE, "green_transfer"),
                            (GREEN, "/World/Cube", "green_lift"),
                            (GREEN, "/World/PlacementBox/Wall", "green_lower"),
                            (BOTTLE, "/World/PlacementBox/Wall", "green_ready"),
                            (BOTTOM, GREEN, "green_transfer")):
            obj = node(phase)
            report(obj, a=a, b=b)
            self.assertTrue(obj.unexpected, (a, b, phase))

    def test_support_cannot_hide_remaining_finger_contact(self):
        obj = node("green_release")
        report(obj, a=BOTTOM, gap=0, impulse=(0, 0, .01))
        report(obj, gap=0, impulse=(0, .01, 0))
        with self.assertRaises(RuntimeError):
            obj.verify_placed()

    def test_recorded_bottle_proximity_requires_separation_and_no_impulse(self):
        obj = node("green_lower")
        report(obj, a=BOTTLE, gap=.01088529359549284)
        self.assertFalse(obj.unexpected)
        self.assertFalse(obj.active_contacts)
        self.assertFalse(obj.loaded_floor_contacts)
        self.assertFalse(obj.bilateral())
        for gap, impulse in ((.00199, (0, 0, 0)), (.01088, (0, 0, 1e-20)), (0, (0, 0, .1))):
            obj = node("green_lower")
            report(obj, a=BOTTLE, gap=gap, impulse=impulse)
            self.assertTrue(obj.unexpected)
        with self.assertRaises(RuntimeError):
            obj.verify_placed()

    def test_other_stage_does_not_clear_or_add_evidence(self):
        obj = node("green_close")
        for finger in (LEFT, RIGHT):
            report(obj, a=finger, gap=0, impulse=(0, .01, 0))
        report(obj, event=ContactEventType.CONTACT_LOST, stage_id=2)
        report(obj, a=PALM, stage_id=2)
        self.assertTrue(obj.bilateral())
        self.assertFalse(obj.unexpected)

    def test_initial_fixture_support_is_allowed(self):
        obj = node("initializing")
        for a, b in ((TABLE, GREEN), (TABLE, "/World/Cube"),
                     (TABLE, "/World/TabletopProps/BlackCube"), (BOTTOM, BOTTLE)):
            report(obj, a=a, b=b, gap=0, impulse=(0, 0, .01))
        self.assertFalse(obj.unexpected)
        self.assertFalse(obj.loaded_floor_contacts)

    def test_parent_red_state_and_force_are_not_overwritten(self):
        forbidden = {"carry_offset", "released_cube", "cube_initial", "cube_quat"}
        assignments = [target for n in ast.walk(CLASS) if isinstance(n, ast.Assign) for target in n.targets]
        self.assertFalse(any(isinstance(t, ast.Attribute) and t.attr in forbidden for t in assignments))
        calls = [n.func.attr for n in ast.walk(CLASS) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden_call in ("CreateMaxForceAttr", "set_world_pose", "set_joint_positions", "Export", "Save"):
            self.assertNotIn(forbidden_call, calls)
        run = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        self.assertEqual(run.body[0].value.func.attr, "preflight")
        self.assertFalse(any(isinstance(n, ast.Try) for n in ast.walk(run)))


class GreenRuntimeGuardsTests(unittest.TestCase):
    def make_node(self, phase="green_hover"):
        obj = node(phase)
        identity = np.array([1., 0., 0., 0.])
        obj.poses = {GREEN: (START.copy(), identity.copy()),
                     BOTTLE: (BOTTLE_XYZ.copy(), identity.copy()),
                     "/World/Cube": (np.array([.5, 0, .025]), identity.copy()),
                     "/World/TabletopProps/BlackCube": (np.array([.73, -.07, 0.]), identity.copy())}
        obj.pose = lambda path: obj.poses[path]
        obj.obstacles = {p: (xyz.copy(), q.copy()) for p, (xyz, q) in obj.poses.items() if p != GREEN}
        obj.green_start, obj.green_rotation = START.copy(), identity.copy()
        obj.carry_offset = obj.released_cube = obj.green_carry_offset = obj.placed_green = None
        obj.green_peak_slip = 0.
        obj.parent_ticks = 0
        obj.parent_fault = False
        obj.fi = [7, 8]
        obj.joints = np.r_[np.zeros(7), .02508, .02508]
        obj.q = lambda: obj.joints
        obj.grasp = lambda: obj.poses[GREEN][0] + [0, 0, .03]
        return obj

    def test_inherited_guards_are_called_and_not_caught(self):
        obj = self.make_node()
        obj.tick()
        self.assertEqual(obj.parent_ticks, 1)
        obj.parent_fault = True
        with self.assertRaisesRegex(RuntimeError, "Inherited"):
            obj.tick()

    def test_red_carry_state_must_stay_unused(self):
        for attr in ("carry_offset", "released_cube"):
            obj = self.make_node()
            setattr(obj, attr, np.zeros(3))
            with self.assertRaisesRegex(RuntimeError, "Parent red-cube"):
                obj.tick()

    def test_green_pregrasp_motion_and_tilt_fail(self):
        obj = self.make_node("green_descend")
        obj.poses[GREEN][0][0] += .0021
        with self.assertRaisesRegex(RuntimeError, "before grasp"):
            obj.tick()
        obj = self.make_node()
        obj.poses[GREEN][1][:] = [np.cos(.101 / 2), 0, 0, np.sin(.101 / 2)]
        with self.assertRaisesRegex(RuntimeError, "tilt"):
            obj.tick()

    def test_all_protected_props_translation_and_rotation(self):
        for path in (BOTTLE, "/World/Cube", "/World/TabletopProps/BlackCube"):
            for mode in ("position", "rotation"):
                obj = self.make_node()
                if mode == "position":
                    obj.poses[path][0][0] += .0031
                else:
                    obj.poses[path][1][:] = [np.cos(.051 / 2), 0, 0, np.sin(.051 / 2)]
                with self.assertRaisesRegex(RuntimeError, "Protected"):
                    obj.tick()

    def test_carry_requires_loaded_bilateral_width_and_four_mm_slip(self):
        for failure in ("none", "bilateral", "width", "slip"):
            obj = self.make_node("green_lift")
            obj.green_carry_offset = np.array([0., 0., -.03])
            for finger in (LEFT, RIGHT):
                report(obj, a=finger, gap=0, impulse=(0, .01, 0))
            if failure == "bilateral":
                report(obj, a=RIGHT, event=ContactEventType.CONTACT_LOST)
            elif failure == "width":
                obj.joints[7] = .02508 + .0016
            elif failure == "slip":
                obj.green_carry_offset[0] = .0041
            if failure == "none":
                obj.tick()
            else:
                with self.assertRaisesRegex(RuntimeError, "grip"):
                    obj.tick()

    def test_low_transfer_and_placed_drift_fail(self):
        obj = self.make_node("green_transfer")
        with self.assertRaisesRegex(RuntimeError, "90 mm"):
            obj.tick()
        obj = self.make_node("green_retreat")
        obj.placed_green = TARGET.copy()
        obj.poses[GREEN][0][:] = TARGET + [0, 0, .0031]
        with self.assertRaisesRegex(RuntimeError, "moved during retreat"):
            obj.tick()


if __name__ == "__main__":
    unittest.main()