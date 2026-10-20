import math

import pytest

from vla_sim.motion import HOME, JOINT_NAMES, LIMITS, demo_target, ordered_positions


def test_reorders_feedback():
    assert ordered_positions(JOINT_NAMES[::-1], HOME[::-1]) == HOME


@pytest.mark.parametrize("positions", [HOME[:-1], (math.nan,) + HOME[1:], (99.0,) + HOME[1:]])
def test_rejects_invalid_feedback(positions):
    with pytest.raises(ValueError):
        ordered_positions(JOINT_NAMES, positions)


def test_waits_for_complete_feedback():
    with pytest.raises(KeyError):
        ordered_positions(["other_joint"], [0.0])


def test_smooth_bounded_motion():
    initial = (0.3,) + HOME[1:]
    assert demo_target(initial, 0.0) == list(initial)
    assert demo_target(initial, 5.0) == list(HOME)
    for step in range(1000):
        target = demo_target(initial, step / 20)
        assert len(target) == 7
        assert all(low <= q <= high for q, (low, high) in zip(target, LIMITS))