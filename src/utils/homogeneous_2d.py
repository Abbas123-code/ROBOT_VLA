"""2D homogeneous transforms — Phase 1, Part B.

Fill in the three functions marked TODO, then run this file:

    python3 src/utils/homogeneous_2d.py

A self-check at the bottom prints PASS/FAIL for each.
Read docs/phase1b_homogeneous.md for the intuition and math.
"""

import math


def transform_matrix(theta, tx, ty):
    """Return the 3x3 homogeneous transform as a list of three rows.

    T = [[cos(theta), -sin(theta), tx],
         [sin(theta),  cos(theta), ty],
         [0,           0,          1 ]]

    theta is in radians.
    """
    # TODO: build and return the 3x3 matrix described above.
    raise NotImplementedError


def apply_transform(T, point):
    """Apply 3x3 transform T to a 2D point = [x, y].

    Steps:
      1. Make the point homogeneous: p = [x, y, 1].
      2. Multiply: p' = T @ p  (a 3-vector).
      3. Return the first two components [x', y'].
    """
    # TODO: implement the three steps above and return [x', y'].
    raise NotImplementedError


def compose(T1, T2):
    """Return the 3x3 matrix product T1 @ T2 (chain two transforms).

    Standard matrix multiply of two 3x3 matrices, by hand:
        result[i][j] = sum over k of T1[i][k] * T2[k][j]
    """
    # TODO: compute and return the 3x3 product as a list of three rows.
    raise NotImplementedError


# ------------------------------------------------------------------
# Self-check. You do not need to edit below this line.
# ------------------------------------------------------------------
def _close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _run(name, fn):
    try:
        ok = fn()
        print(f"[test] {name:28s} -> {'PASS' if ok else 'FAIL'}")
        return ok
    except NotImplementedError:
        print(f"[test] {name:28s} -> FAIL (not implemented)")
        return False


def _main():
    passed = True

    def t_identity():
        T = transform_matrix(0.0, 0.0, 0.0)
        return (_close(T[0], [1.0, 0.0, 0.0])
                and _close(T[1], [0.0, 1.0, 0.0])
                and _close(T[2], [0.0, 0.0, 1.0]))

    def t_translation():
        T = transform_matrix(0.0, 1.0, 0.0)
        return _close(apply_transform(T, [0.4, 0.1]), [1.4, 0.1])

    def t_rotation():
        T = transform_matrix(math.pi / 2, 0.0, 0.0)
        return _close(apply_transform(T, [1.0, 0.0]), [0.0, 1.0])

    def t_rot_trans():
        T = transform_matrix(math.pi / 2, 1.0, 0.0)
        return _close(apply_transform(T, [1.0, 0.0]), [1.0, 1.0])

    def t_compose():
        # T1 translates by (1,0); T2 rotates 90deg. Right matrix applies first.
        T1 = transform_matrix(0.0, 1.0, 0.0)
        T2 = transform_matrix(math.pi / 2, 0.0, 0.0)
        T = compose(T1, T2)
        # rotate [1,0] -> [0,1], then translate +[1,0] -> [1,1]
        return _close(apply_transform(T, [1.0, 0.0]), [1.0, 1.0])

    passed &= _run("transform_matrix identity", t_identity)
    passed &= _run("pure translation", t_translation)
    passed &= _run("pure rotation 90deg", t_rotation)
    passed &= _run("rotation + translation", t_rot_trans)
    passed &= _run("compose (chain frames)", t_compose)

    print("All tests passed. 🎉" if passed else "Some tests failed — see above.")


if __name__ == "__main__":
    _main()
