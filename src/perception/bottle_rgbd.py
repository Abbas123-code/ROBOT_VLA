"""Model-specific cyan bottle detector. No USD, object poses or scene access.

Assumptions: upright opaque cyan bottle with blue label, 64 mm cylindrical
body, on a horizontal table at world Z=0. This is not a general bottle model.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


def locate_bottle(rgb, depth, k, camera_to_world):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 image")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Registered RGB and depth required")
    k = np.asarray(k, dtype=float).reshape(3, 3)
    transform = np.asarray(camera_to_world, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Finite optical-camera transform required")
    y, x = np.indices(depth.shape)
    points = np.stack(((x - k[0, 2]) * depth / k[0, 0],
                       (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
    with np.errstate(invalid="ignore"):
        world = points @ transform[:3, :3].T + transform[:3, 3]
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    # Cyan/blue, excluding grey, green cubes and orange objects.
    mask = (b > r * 1.15) & (g > r * 1.10) & (b > 90) & (g > 75)
    mask &= np.isfinite(depth) & (depth > .02) & (depth < 5)
    labels, count = ndimage.label(mask)
    candidates = []
    for label in range(1, count + 1):
        selected = labels == label
        if selected.sum() < 60:
            continue
        p = world[selected]
        # Reject the low receiving box; a bottle must have a tall cyan body.
        lo, hi = np.percentile(p[:, 2], [5, 95])
        if lo > .09 or hi < .12 or hi > .205 or hi - lo < .07:
            continue
        # Fit only the cylindrical body, not neck/shoulder or cap.
        body = p[(p[:, 2] > .025) & (p[:, 2] < .12)]
        if len(body) < 50:
            continue
        xy = body[:, :2]
        result = least_squares(lambda c: np.linalg.norm(xy - c, axis=1) - .03225,
                               np.mean(xy, axis=0), loss="soft_l1", f_scale=.002)
        residual = np.abs(np.linalg.norm(xy - result.x, axis=1) - .03225)
        inliers = residual < .004
        if inliers.mean() < .85 or np.median(residual) > .002:
            continue
        angles = np.arctan2(*(xy[inliers] - result.x).T[::-1])
        # Visible arc must constrain the centre, not just one small surface patch.
        coverage = 2 * np.pi - np.max(np.diff(np.r_[np.sort(angles), np.sort(angles)[0] + 2*np.pi]))
        if coverage < .8:
            continue
        rows, cols = np.where(selected)
        candidates.append({"center_xy": result.x.tolist(), "radius_m": .03225,
                           "body_points": len(body), "fit_median_error_m": float(np.median(residual)),
                           "inlier_fraction": float(inliers.mean()), "visible_arc_rad": float(coverage),
                           "visible_height_range_m": [float(lo), float(hi)],
                           "bbox_xyxy": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
                           "method": "RGB cyan component + registered depth + known-radius cylinder fit"})
    if len(candidates) != 1:
        raise ValueError(f"Expected one unambiguous bottle; found {len(candidates)}")
    return candidates[0]