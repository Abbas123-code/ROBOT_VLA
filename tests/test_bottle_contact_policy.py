"""Exercise real controller methods without importing Isaac or starting physics.

Abort assertions check ``unexpected``: contacts() records the fault, and the
production tick guard raises later. No controller policy is reimplemented here.
"""
import ast
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.contact_evidence import classify_contact

BOTTLE = "/World/TabletopProps/WaterBottle"
LEFT = "/World/panda/panda_leftfinger"
RIGHT = "/World/panda/panda_rightfinger"
PALM = "/World/panda/panda_hand"
ContactEventType = SimpleNamespace(CONTACT_FOUND=1, CONTACT_PERSIST=2, CONTACT_LOST=3)
PhysicsSchemaTools = SimpleNamespace(intToSdfPath=lambda value: value)

SOURCE = ROOT / "isaac/scripts/yolo_bottle_place_controller.py"
controller = next(node for node in ast.parse(SOURCE.read_text()).body
                  if isinstance(node, ast.ClassDef) and node.name == "YOLOBottlePlaceNode")
methods = [node for node in controller.body
           if isinstance(node, ast.FunctionDef) and node.name in ("contacts", "bilateral")]
assert {node.name for node in methods} == {"contacts", "bilateral"}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), globals())


def make_node(phase="bottle_descend"):
    node = SimpleNamespace(
        stage_id=7, phase=phase, paths={"panda_link0": "/World/panda/panda_link0"},
        fingers={LEFT, RIGHT}, active_contacts=set(), loaded_finger_contacts=set(),
        loaded_floor_contacts=set(),
        unexpected=set(), proximity_events=0,
    )
    node.contacts = MethodType(globals()["contacts"], node)
    node.bilateral = MethodType(globals()["bilateral"], node)
    return node


def report(node, actor=LEFT, collider=BOTTLE + "/Body", gap=.002,
           impulse=(0., -0., 0.), event=ContactEventType.CONTACT_FOUND,
           stage_id=7, reverse=False):
    actors = (actor, BOTTLE)
    colliders = (actor + "/Collision", collider)
    if reverse:
        actors, colliders = actors[::-1], colliders[::-1]
    header = SimpleNamespace(
        stage_id=stage_id, actor0=actors[0], actor1=actors[1],
        collider0=colliders[0], collider1=colliders[1], type=event,
        contact_data_offset=0, num_contact_data=int(event != ContactEventType.CONTACT_LOST),
    )
    data = [] if not header.num_contact_data else [SimpleNamespace(separation=gap, impulse=impulse)]
    node.contacts([header], data)
    return (tuple(sorted(actors)), tuple(sorted(colliders)))


class BottleContactPolicyTests(unittest.TestCase):
    def test_table_under_tray_proximity_never_counts_as_support(self):
        table = "/World/table/Geometry/baseLink/box"
        node = make_node("bottle_lower")
        report(node, actor=table, gap=.01)
        self.assertFalse(node.unexpected)
        self.assertFalse(node.loaded_floor_contacts)
        self.assertFalse(node.active_contacts)
        for gap, impulse in ((.01, (1e-20, 0, 0)), (.001, (0, 0, 0)), (0., (0, 0, .01))):
            node = make_node("bottle_lower")
            report(node, actor=table, gap=gap, impulse=impulse)
            self.assertTrue(node.unexpected)

    def test_floor_support_requires_load_and_clears_on_lost(self):
        floor = "/World/PlacementBox/Bottom"
        node = make_node("bottle_release")
        report(node, actor=floor, gap=.01)
        self.assertFalse(node.loaded_floor_contacts)
        key = report(node, actor=floor, gap=0., impulse=(0, 0, .01))
        self.assertEqual(node.loaded_floor_contacts, {key})
        node.phase = "return_home"
        report(node, actor=floor, gap=0., impulse=(0, 0, .01))
        self.assertFalse(node.unexpected)
        report(node, actor=floor, event=ContactEventType.CONTACT_LOST)
        self.assertFalse(node.loaded_floor_contacts)

    def test_recorded_shoulder_proximity_is_not_touch_or_support(self):
        node = make_node()
        report(node, collider=BOTTLE + "/Shoulder", gap=.01074875146150589)
        self.assertEqual(node.proximity_events, 1)
        self.assertFalse(node.unexpected)
        self.assertFalse(node.active_contacts)
        self.assertFalse(node.bilateral())
        for gap, impulse in ((.01, (1e-20, 0, 0)), (.001, (0, 0, 0)), (0., (0, .01, 0))):
            node = make_node()
            report(node, collider=BOTTLE + "/Shoulder", gap=gap, impulse=impulse)
            self.assertTrue(node.unexpected)

    def test_exact_finger_body_two_mm_zero_impulse_descending_is_ignored(self):
        for finger in (LEFT, RIGHT):
            for reverse in (False, True):
                with self.subTest(finger=finger, reverse=reverse):
                    node = make_node()
                    report(node, actor=finger, reverse=reverse)
                    self.assertEqual(node.proximity_events, 1)
                    self.assertFalse(node.active_contacts)
                    self.assertFalse(node.loaded_finger_contacts)
                    self.assertFalse(node.unexpected)
                    self.assertFalse(node.bilateral())

    def test_palm_and_non_body_zero_impulse_are_never_ignored(self):
        for actor, collider in ((PALM, BOTTLE + "/Body"),
                                (LEFT, BOTTLE + "/Neck"),
                                (LEFT, BOTTLE + "/BodyExtra"),
                                (LEFT, BOTTLE + "/Body/Child")):
            with self.subTest(actor=actor, collider=collider):
                node = make_node()
                key = report(node, actor=actor, collider=collider)
                self.assertIn(key, node.active_contacts)
                self.assertTrue(node.unexpected)
                self.assertEqual(node.proximity_events, 0)
                self.assertFalse(node.loaded_finger_contacts)

    def test_body_proximity_during_hover_is_not_ignored(self):
        node = make_node("bottle_hover")
        key = report(node)
        self.assertIn(key, node.active_contacts)
        self.assertTrue(node.unexpected)
        self.assertEqual(node.proximity_events, 0)

    def test_nonzero_impulse_at_positive_gap_flags_abort(self):
        for impulse in ((1e-20, 0., 0.), (0., -.01, 0.), (0., 0., .01)):
            with self.subTest(impulse=impulse):
                node = make_node()
                key = report(node, impulse=impulse)
                self.assertIn(key, node.active_contacts)
                self.assertTrue(node.unexpected)
                self.assertFalse(node.loaded_finger_contacts)
                self.assertEqual(node.proximity_events, 0)

    def test_actual_touch_during_descent_flags_abort_even_without_impulse(self):
        for gap in (0., -.001):
            for impulse in ((0., 0., 0.), (0., .01, 0.)):
                with self.subTest(gap=gap, impulse=impulse):
                    node = make_node()
                    key = report(node, gap=gap, impulse=impulse)
                    self.assertIn(key, node.active_contacts)
                    self.assertTrue(node.unexpected)
                    self.assertEqual(node.proximity_events, 0)

    def test_only_loaded_touch_on_both_fingers_counts_as_grip(self):
        node = make_node("body_close")
        self.assertFalse(node.bilateral())
        left = report(node, gap=0., impulse=(0., .01, 0.))
        self.assertEqual(node.loaded_finger_contacts, {left})
        self.assertFalse(node.bilateral())
        report(node, actor=RIGHT, gap=0.)  # Touch without load is not grip.
        self.assertFalse(node.bilateral())
        report(node, actor=RIGHT, gap=.002, impulse=(0., .01, 0.))
        self.assertFalse(node.bilateral())  # Speculative force is not support.
        right = report(node, actor=RIGHT, gap=.0005, impulse=(0., .01, 0.))
        self.assertEqual(node.loaded_finger_contacts, {left, right})
        self.assertTrue(node.bilateral())
        self.assertFalse(node.unexpected)
        node.contacts([], [])  # Sleeping-body silence preserves loaded contact.
        self.assertTrue(node.bilateral())

    def test_proximity_on_both_fingers_is_not_bilateral(self):
        for phase in ("bottle_descend", "body_close"):
            with self.subTest(phase=phase):
                node = make_node(phase)
                keys = {report(node, actor=finger) for finger in (LEFT, RIGHT)}
                self.assertFalse(node.loaded_finger_contacts)
                self.assertFalse(node.bilateral())
                self.assertFalse(node.unexpected)
                self.assertEqual(node.active_contacts, keys if phase == "body_close" else set())

    def test_lost_clears_active_and_loaded_contact_without_points(self):
        node = make_node("body_close")
        keys = {finger: report(node, actor=finger, gap=0., impulse=(0., .01, 0.))
                for finger in (LEFT, RIGHT)}
        self.assertTrue(node.bilateral())
        report(node, event=ContactEventType.CONTACT_LOST, reverse=True)
        self.assertEqual(node.active_contacts, {keys[RIGHT]})
        self.assertEqual(node.loaded_finger_contacts, {keys[RIGHT]})
        self.assertFalse(node.bilateral())
        report(node, actor=RIGHT, event=ContactEventType.CONTACT_LOST)
        self.assertFalse(node.active_contacts)
        self.assertFalse(node.loaded_finger_contacts)
        self.assertFalse(node.unexpected)

    def test_loaded_to_proximity_clears_stale_grip(self):
        for phase in ("bottle_descend", "body_close"):
            with self.subTest(phase=phase):
                node = make_node("body_close")
                keys = {finger: report(node, actor=finger, gap=0., impulse=(0., .01, 0.))
                        for finger in (LEFT, RIGHT)}
                self.assertTrue(node.bilateral())
                node.phase = phase
                report(node, event=ContactEventType.CONTACT_PERSIST, reverse=True)
                self.assertEqual(node.loaded_finger_contacts, {keys[RIGHT]})
                self.assertFalse(node.bilateral())
                expected = {keys[RIGHT]} if phase == "bottle_descend" else set(keys.values())
                self.assertEqual(node.active_contacts, expected)
                self.assertEqual(node.proximity_events, int(phase == "bottle_descend"))
                self.assertFalse(node.unexpected)

    def test_wrong_stage_ignores_found_persist_and_lost(self):
        node = make_node()
        report(node, stage_id=8)
        report(node, actor=PALM, gap=0., impulse=(0., .01, 0.), stage_id=8)
        self.assertFalse(node.active_contacts)
        self.assertFalse(node.loaded_finger_contacts)
        self.assertFalse(node.unexpected)
        self.assertEqual(node.proximity_events, 0)
        node.phase = "body_close"
        keys = {report(node, actor=finger, gap=0., impulse=(0., .01, 0.))
                for finger in (LEFT, RIGHT)}
        for event in (ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST):
            with self.subTest(event=event):
                report(node, event=event, stage_id=8)
                self.assertEqual(node.active_contacts, keys)
                self.assertEqual(node.loaded_finger_contacts, keys)
                self.assertTrue(node.bilateral())
                self.assertFalse(node.unexpected)
                self.assertEqual(node.proximity_events, 0)


if __name__ == "__main__":
    unittest.main()