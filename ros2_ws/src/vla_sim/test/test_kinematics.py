import math

import numpy as np
import pytest

from vla_sim.kinematics import link_transform, rotation


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "test.urdf"
    path.write_text('''<robot name="test">
      <joint name="turn" type="revolute"><parent link="base"/><child link="arm"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="offset" type="fixed"><parent link="arm"/><child link="tool"/>
        <origin xyz="1 0 0" rpy="0 0 0"/></joint>
      <joint name="slide" type="prismatic"><parent link="tool"/><child link="finger"/>
        <axis xyz="1 0 0"/></joint>
    </robot>''')
    return path


def test_rotated_offset(model):
    transform = link_transform(model, {"turn": math.pi / 2}, "tool", "base")
    np.testing.assert_allclose(transform[:3, 3], [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)


def test_prismatic_axis_is_local(model):
    transform = link_transform(model, {"turn": math.pi / 2, "slide": 0.5}, "finger", "base")
    np.testing.assert_allclose(transform[:3, 3], [1, 1.5, 0], atol=1e-12)


def test_missing_feedback(model):
    with pytest.raises(KeyError):
        link_transform(model, {}, "tool", "base")


def test_missing_link(model):
    with pytest.raises(ValueError):
        link_transform(model, {}, "unknown", "base")


@pytest.mark.parametrize("angle", [math.nan, math.inf])
def test_nonfinite_feedback(model, angle):
    with pytest.raises(ValueError):
        link_transform(model, {"turn": angle}, "tool", "base")


def test_zero_axis_rejected():
    with pytest.raises(ValueError):
        rotation([0, 0, 0], 1.0)