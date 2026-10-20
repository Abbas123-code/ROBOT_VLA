import math

import pytest

from vla_sim.approach import check_grasp_position, sample


def test_endpoints_and_clamping():
    plan = {"duration_seconds": 12, "start_arm_radians": [0.1] * 7, "target_arm_radians": [0.5] * 7}
    assert sample(plan, -1) == [0.1] * 7
    assert sample(plan, 0) == [0.1] * 7
    assert sample(plan, 6) == pytest.approx([0.3] * 7)
    assert sample(plan, 12) == [0.5] * 7
    assert sample(plan, 20) == [0.5] * 7
    with pytest.raises(ValueError):
        sample(plan, math.nan)


@pytest.mark.parametrize("position", [[0.5, 0, 0.069], [0.509, 0, 0.08], [math.nan, 0, 0.08]])
def test_descent_bounds_reject_bad_pose(position):
    plan = {"target_world_xyz": [0.5, 0, 0.08], "minimum_grasp_world_z_metres": 0.07,
            "maximum_grasp_lateral_offset_metres": 0.008}
    with pytest.raises(ValueError):
        check_grasp_position(plan, position)


def test_descent_bounds_accept_clear_aligned_pose():
    plan = {"target_world_xyz": [0.5, 0, 0.08], "minimum_grasp_world_z_metres": 0.07,
            "maximum_grasp_lateral_offset_metres": 0.008}
    check_grasp_position(plan, [0.498, 0.0001, 0.075])