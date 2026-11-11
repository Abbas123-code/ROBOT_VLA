"""Regression tests for actual separated, zero-impulse PhysX reports."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.contact_evidence import classify_contact


class ContactEvidenceTests(unittest.TestCase):
    def test_recorded_event_is_proximity_not_grip(self):
        self.assertEqual(classify_contact([.010751154273748398], [[0., -0., 0.]]), "proximity")

    def test_proximity_requires_two_mm_margin(self):
        self.assertEqual(classify_contact([.002], [[0, 0, 0]]), "proximity")
        for gap in (.001999, .0005, 0., -.001):
            self.assertEqual(classify_contact([gap], [[0, 0, 0]]), "guarded")

    def test_any_speculative_impulse_is_guarded(self):
        for impulse in (1e-20, .01, -1.):
            self.assertEqual(classify_contact([.01], [[impulse, 0, 0]]), "guarded")

    def test_support_requires_near_touch_and_impulse_same_point(self):
        self.assertEqual(classify_contact([.0005], [[0, .01, 0]]), "loaded_touch")
        self.assertEqual(classify_contact([-.001], [[0, .01, 0]]), "loaded_touch")
        self.assertEqual(classify_contact([0., .01], [[0, 0, 0], [0, .01, 0]]), "guarded")

    def test_every_point_must_be_separated(self):
        self.assertEqual(classify_contact([.01, .001], [[0, 0, 0], [0, 0, 0]]), "guarded")

    def test_invalid_evidence_fails_closed(self):
        for gaps, impulses in (([], []), ([.01], []), ([np.nan], [[0, 0, 0]]),
                               ([.01], [[np.inf, 0, 0]]), ([.01], [0, 0, 0]),
                               ([[.01]], [[0, 0, 0]]), (["bad"], [[0, 0, 0]])):
            with self.subTest(gaps=gaps, impulses=impulses):
                self.assertEqual(classify_contact(gaps, impulses), "guarded")


if __name__ == "__main__":
    unittest.main()