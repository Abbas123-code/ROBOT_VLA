"""Reconstructed fixture geometry and nominal IK preflight; no simulator writes.

GreenCube's root is its BOTTOM, not its centre. These are simulator-known
fixtures, not detections. A passing IK preflight does not prove collision safety.
"""
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

GREEN = "/World/TabletopProps/GreenCube"
BOTTLE = "/World/TabletopProps/WaterBottle"
START = np.array([.70, .14, 0.])
TARGET = np.array([.512, -.284, .008])
TRAY_XY = np.array([.45, -.25])
FLOOR = .008
INNER_HALF = .092
DOWN = np.diag([1., -1., -1.])
CENTRAL = np.array([.5, .23, .35])
# Positive measured clearance, not a claim that the full allowed tilt/slip
# envelope fits. Actual poses must pass on EVERY tick. Nominal bottle gap is
# 8.3007 mm and rim gap is 5 mm, so a 10 mm clearance claim would be false.
MIN_CLEARANCE = .002


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Finite {size}-vector required")
    return result.copy()


def rotation(quaternion):
    """USD wxyz -> orthonormal matrix; reject malformed physics evidence."""
    q = vector(quaternion, 4)
    if abs(np.linalg.norm(q) - 1.) > 1e-4:
        raise ValueError("Unit quaternion required")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def angle_between(a, b):
    return float(Rotation.from_matrix(rotation(a).T @ rotation(b)).magnitude())


def bounds(root, quaternion, bottle=False):
    """World AABB: exact cube, conservative enclosing bottle prism.

    Bottle collision geometry lies within radius .032, z=[0,.202]. The
    enclosing prism remains conservative at nonzero bottle tilt.
    """
    r = .032 if bottle else .025
    height = .202 if bottle else .05
    corners = np.array(list(product((-r, r), (-r, r), (0., height))))
    points = corners @ rotation(quaternion).T + vector(root)
    return points.min(axis=0), points.max(axis=0)


def placement_geometry(bottle_xyz, bottle_quat=(1., 0., 0., 0.),
                       green_xyz=TARGET, green_quat=(1., 0., 0., 0.)):
    """Fail closed on tray overflow or occupied bottle footprint.

    XY AABB separation is intentionally conservative even for rotated objects.
    Height is checked separately for release/support by the controller.
    """
    bottle_xyz = vector(bottle_xyz)
    if abs(bottle_xyz[2] - FLOOR) > .002:
        raise ValueError("Reconstructed bottle root is not on the tray floor")
    if angle_between(bottle_quat, (1., 0., 0., 0.)) > .05:
        raise ValueError("Reconstructed bottle is not upright")
    lo, hi = bounds(green_xyz, green_quat)
    blo, bhi = bounds(bottle_xyz, bottle_quat, bottle=True)
    rim = float(min(np.min(lo[:2] - (TRAY_XY - INNER_HALF)),
                    np.min((TRAY_XY + INNER_HALF) - hi[:2])))
    bottle_rim = float(min(np.min(blo[:2] - (TRAY_XY - INNER_HALF)),
                           np.min((TRAY_XY + INNER_HALF) - bhi[:2])))
    gap = float(np.max(np.maximum(lo[:2] - bhi[:2], blo[:2] - hi[:2])))
    if rim < MIN_CLEARANCE or bottle_rim < MIN_CLEARANCE or gap < MIN_CLEARANCE:
        raise ValueError(f"Occupied/unsafe tray: rim={rim}, bottle_rim={bottle_rim}, bottle_gap={gap}")
    return {"rim_clearance_m": rim, "bottle_clearance_m": gap,
            "green_bottom_z": float(lo[2])}


def nominal_goals(planner, home, base=(0., 0., 0.)):
    """Exact command-space route, including orientation changes and HOME."""
    frame = planner.fk(vector(home, 7))
    home_rotation = frame[:3, :3]
    nominal_offset = START - np.array([.7, .14, .030])
    lower = TARGET + [0., 0., .002] - nominal_offset
    return [
        ("green_hover", CENTRAL.copy(), home_rotation),
        ("green_hover", CENTRAL.copy(), DOWN),
        ("green_hover", np.array([.7, .14, .32]), DOWN),
        ("green_descend", np.array([.7, .14, .030]), DOWN),
        ("green_lift", np.array([.7, .14, .32]), DOWN),
        ("green_transfer", np.array([.5, .14, .32]), DOWN),
        ("green_transfer", np.r_[TARGET[:2], .32], DOWN),
        ("green_lower", lower, DOWN),
        ("green_retreat", np.r_[lower[:2], .32], DOWN),
        ("green_home", CENTRAL.copy(), DOWN),
        ("green_home", CENTRAL.copy(), home_rotation),
        ("green_home", frame[:3, 3] + vector(base), home_rotation),
    ]


def preflight(planner, home, bottle_xyz, base=(0., 0., 0.)):
    """Full nominal Cartesian path + bounded exact HOME correction, no motion."""
    geometry = placement_geometry(bottle_xyz)
    seed = vector(home, 7)
    goals = nominal_goals(planner, home, base)
    peak = 0.
    for phase, point, orientation in goals:
        samples = planner.segment(seed, point - vector(base), orientation)
        if not np.isfinite(samples).all():
            raise RuntimeError(f"Non-finite nominal IK: {phase}")
        peak = max(peak, float(np.max(np.abs(np.diff(samples, axis=0)))))
        seed = samples[-1]
    correction = float(np.max(np.abs(vector(home, 7) - seed)))
    if correction > .15:
        raise RuntimeError(f"Nominal HOME correction exceeds .15 rad: {correction}")
    return {"segments": len(goals), "peak_branch_step_rad": peak,
            "home_correction_rad": correction, **geometry,
            "scope": "Kinematics and fixture geometry only; not collision certification"}