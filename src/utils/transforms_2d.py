"""2D transforms — Phase 1, Part A.

Fill in the three functions marked TODO, then run this file:

    python3 src/utils/transforms_2d.py

A self-check at the bottom prints PASS/FAIL for each function.
Read docs/phase1_transforms.md for the intuition and math.
"""

import math


def translate(point, offset):
    """Return point shifted by offset.

    point  = [x, y]
    offset = [tx, ty]
    result = [x + tx, y + ty]
    """
    # TODO: return the element-wise sum of point and offset.
    return [point[0] + offset[0], point[1] + offset[1]]


def rotation_matrix(theta):
    """Return the 2x2 rotation matrix R(theta) as a list of two rows.

    R = [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]

    theta is in radians. Use math.cos / math.sin.
    """
    # TODO: build and return the 2x2 matrix described above.
    return [[math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]]


def rotate(point, theta):
    """Rotate point = [x, y] by theta radians (counter-clockwise).

    Apply the rotation matrix:
        x' = cos(theta)*x - sin(theta)*y
        y' = sin(theta)*x + cos(theta)*y
    Return [x', y'].
    """
    # TODO: use rotation_matrix(theta) (or the formulas above) and return [x', y'].
    return [math.cos(theta) * point[0] - math.sin(theta) * point[1],
            math.sin(theta) * point[0] + math.cos(theta) * point[1]]



# ------------------------------------------------------------------
# Self-check. You do not need to edit below this line.
# ------------------------------------------------------------------
def _close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _main():
    passed = True

    try:
        ok = _close(translate([0.4, 0.1], [1.0, 0.0]), [1.4, 0.1])
        print(f"[test] translate       -> {'PASS' if ok else 'FAIL'}")
        passed &= ok
    except NotImplementedError:
        print("[test] translate       -> FAIL (not implemented)")
        passed = False

    try:
        R = rotation_matrix(0.0)
        ok = _close(R[0], [1.0, 0.0]) and _close(R[1], [0.0, 1.0])
        print(f"[test] rotation_matrix -> {'PASS' if ok else 'FAIL'}")
        passed &= ok
    except NotImplementedError:
        print("[test] rotation_matrix -> FAIL (not implemented)")
        passed = False

    try:
        ok = _close(rotate([1.0, 0.0], math.pi / 2), [0.0, 1.0])
        print(f"[test] rotate 90deg    -> {'PASS' if ok else 'FAIL'}")
        passed &= ok
    except NotImplementedError:
        print("[test] rotate 90deg    -> FAIL (not implemented)")
        passed = False

    try:
        ok = _close(rotate([2.0, -3.0], 0.0), [2.0, -3.0])
        print(f"[test] rotate 0deg     -> {'PASS' if ok else 'FAIL'}")
        passed &= ok
    except NotImplementedError:
        print("[test] rotate 0deg     -> FAIL (not implemented)")
        passed = False

    print("All tests passed. 🎉" if passed else "Some tests failed — see above.")


if __name__ == "__main__":
    _main()
