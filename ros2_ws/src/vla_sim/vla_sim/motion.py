import math

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
HOME = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)
LIMITS = ((-2.9671, 2.9671), (-1.8326, 1.8326), (-2.9671, 2.9671),
          (-3.1416, 0.0), (-2.9671, 2.9671), (-0.0873, 3.8223), (-2.9671, 2.9671))


def ordered_positions(names, positions):
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("Joint names and positions must be unique and equally sized")
    mapping = dict(zip(names, positions))
    values = tuple(float(mapping[name]) for name in JOINT_NAMES)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite joint position")
    if any(not low <= value <= high for value, (low, high) in zip(values, LIMITS)):
        raise ValueError("Joint position is outside model limits")
    return values


def demo_target(initial, elapsed):
    elapsed = max(0.0, elapsed)
    fraction = min(elapsed / 5.0, 1.0)
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    target = [start + blend * (end - start) for start, end in zip(initial, HOME)]
    if elapsed > 5.0:
        phase = elapsed - 5.0
        ramp = min(phase / 3.0, 1.0)
        target[0] += 0.15 * ramp * ramp * (3.0 - 2.0 * ramp) * math.sin(phase * 0.5)
    return target