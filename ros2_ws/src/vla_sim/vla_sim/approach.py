import hashlib
import json
import math

from vla_sim.motion import JOINT_NAMES, ordered_positions


def load_plan(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    for key in ("start_arm_radians", "target_arm_radians"):
        ordered_positions(JOINT_NAMES, plan[key])
    duration = float(plan["duration_seconds"])
    settle = float(plan["settle_seconds"])
    if not math.isfinite(duration) or not 5 <= duration <= 60:
        raise ValueError("Approach duration must be between 5 and 60 seconds")
    if not math.isfinite(settle) or not 1 <= settle <= 10:
        raise ValueError("Settling duration must be between 1 and 10 seconds")
    if max(abs(b - a) * 1.5 / duration for a, b in zip(
        plan["start_arm_radians"], plan["target_arm_radians"]
    )) > 0.2:
        raise ValueError("Approach exceeds the 0.2 rad/s demonstration speed limit")
    return plan, hashlib.sha256(raw).hexdigest()


def check_grasp_position(plan, position):
    """Check optional descent bounds for a measured or explicitly estimated pose."""
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("Non-finite or invalid grasp position")
    minimum = plan.get("minimum_grasp_world_z_metres")
    if minimum is not None and position[2] < minimum:
        raise ValueError("Grasp point crossed the descent height limit")
    lateral = plan.get("maximum_grasp_lateral_offset_metres")
    if lateral is not None and math.dist(position[:2], plan["target_world_xyz"][:2]) > lateral:
        raise ValueError("Grasp point moved outside the descent alignment limit")


def sample(plan, elapsed):
    if not math.isfinite(elapsed):
        raise ValueError("Non-finite simulation time")
    fraction = min(max(elapsed / plan["duration_seconds"], 0.0), 1.0)
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    return [a + blend * (b - a) for a, b in zip(
        plan["start_arm_radians"], plan["target_arm_radians"]
    )]