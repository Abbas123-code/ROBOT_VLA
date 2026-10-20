"""Kinematic checks only; passing does not certify collision-free motion."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ros2_ws/src/vla_sim")]
from src.control.bottle_grasp_plan import BottlePosePlanner, body_orientation
from vla_sim.kinematics import link_transform
from vla_sim.motion import HOME, JOINT_NAMES, LIMITS


def main():
    urdf = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
    planner = BottlePosePlanner(urdf, JOINT_NAMES, LIMITS)
    bounds = np.asarray(LIMITS)
    for angles in np.random.default_rng(1).uniform(bounds[:, 0], bounds[:, 1], (20, 7)):
        np.testing.assert_allclose(planner.fk(angles), link_transform(urdf, dict(zip(JOINT_NAMES, angles))), atol=1e-12)
    xy = np.array([.7224538456, .3420420148])
    rotation = body_orientation(xy, 45, -1)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.)
    body = np.r_[xy, .13]
    hover = np.array([.5, .23, .35])
    seed = np.asarray(HOME)
    target = np.array([.45, -.25, .142])
    goals = [(hover, planner.fk(seed)[:3, :3]), (hover, rotation),
             (body - .14 * rotation[:, 2], rotation), (body, rotation),
             (body + [0, 0, .015], rotation), (np.r_[xy, .27], rotation),
             (np.array([.5, .2, .27]), rotation), (np.array([.45, -.25, .27]), rotation),
             (target, rotation), (target - .10 * rotation[:, 2], rotation)]
    peak = 0.
    for position, orientation in goals:
        samples = planner.segment(seed, position, orientation)
        peak = max(peak, float(np.max(np.abs(np.diff(samples, axis=0)))))
        seed = samples[-1]
        np.testing.assert_allclose(planner.fk(seed)[:3, 3], position, atol=1e-4)
    report = {"completed": True, "scope": "Kinematics only; no physics or collision validation",
              "fk_reference_samples": 20, "route_segments": len(goals), "peak_branch_step_rad": peak}
    (ROOT / "ros2_ws/log/bottle_pose_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BOTTLE_POSE_PLAN_PASSED " + json.dumps(report))


if __name__ == "__main__":
    main()