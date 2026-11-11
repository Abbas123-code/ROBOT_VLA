"""Refine independently detected bottle/tray ROIs using registered RGB-D.

Restricted geometry: upright cyan cylinder on Z=0, axis-aligned blue tray with
200mm exterior and 8mm floor. No USD imports, object-pose lookup or XY fallback.
"""
import numpy as np
from scipy import ndimage

from perception.bottle_rgbd import locate_bottle


def unique_target(objects, label, class_id):
    matches = [o for o in objects if o["label"] == label and o["class_id"] == class_id and o["confidence"] >= .5]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one {label} >=0.5, found {len(matches)}")
    return matches[0]


def roi_mask(shape, box):
    x1, y1, x2, y2 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y1):max(0, min(shape[0], y2+1)), max(0, x1):max(0, min(shape[1], x2+1))] = True
    return mask


def world_points(depth, k, transform):
    k = np.asarray(k, dtype=float).reshape(3, 3)
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid calibration/transform")
    y, x = np.indices(depth.shape)
    p = np.stack(((x-k[0, 2])*depth/k[0, 0], (y-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    with np.errstate(invalid="ignore"):
        return p @ transform[:3, :3].T + transform[:3, 3]


def locate_tray(rgb, depth, k, transform, target):
    world = world_points(depth, k, transform)
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    mask = roi_mask(depth.shape, target["bbox_xyxy"])
    mask &= (b > r*1.25) & (b > g*1.12) & (g > r*1.3) & (b > 55)
    mask &= np.isfinite(world).all(axis=2) & (world[:, :, 2] > .002) & (world[:, :, 2] < .07)
    labels, n = ndimage.label(mask)
    if not n:
        raise ValueError("No blue tray depth surface inside detection")
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    points = world[labels == sizes.argmax()]
    if len(points) < 80:
        raise ValueError("Too few tray depth points")
    # Visible rim reaches the full outer extent even when the inside floor is occluded.
    low, high = np.percentile(points[:, :2], [1, 99], axis=0)
    size = high-low
    if np.any(size < .175) or np.any(size > .215):
        raise ValueError(f"Tray extent inconsistent with known 200mm geometry: {size}")
    center = (low+high)/2
    inside = np.all(np.abs(points[:, :2]-center) < .075, axis=1)
    floor_points = points[inside & (points[:, 2] < .025)]
    if len(floor_points) < 25:
        raise ValueError("Tray floor not visible; cannot confirm an empty placement area")
    floor = float(np.median(floor_points[:, 2]))
    if abs(floor-.008) > .003 or np.percentile(np.abs(floor_points[:, 2]-floor), 90) > .002:
        raise ValueError("Tray floor not flat at expected height")
    # Check ALL visible depth in the central placement area, not only blue pixels.
    placement = np.all(np.abs(world[:, :, :2]-center) < .065, axis=2) & roi_mask(depth.shape, target["bbox_xyxy"])
    if np.count_nonzero(placement & (world[:, :, 2] > floor+.015)) > 5:
        raise ValueError("Tray placement area contains an obstacle")
    return {"center_xy": center.tolist(), "floor_z": floor, "rim_z": .06,
            "visible_extent_xy": size.tolist(), "floor_points": len(floor_points),
            "method": "YOLO ROI + blue RGB-D extent + flat empty floor; known axis-aligned tray"}


def locate_targets(rgb, depth, k, transform, objects):
    bottle = unique_target(objects, "Water Bottle", 6)
    tray = unique_target(objects, "Blue Tray", 0)
    roi = np.zeros_like(rgb)
    selected = roi_mask(depth.shape, bottle["bbox_xyxy"])
    roi[selected] = rgb[selected]
    estimate = locate_bottle(roi, depth, k, transform)
    receiver = locate_tray(rgb, depth, k, transform, tray)
    estimate.update(detector="WongKinYiu/yolov9", yolo_bottle=bottle, yolo_tray=tray,
                    yolo_objects=objects, tray_center_xy=receiver["center_xy"],
                    tray_floor_z=receiver["floor_z"], tray_rim_z=receiver["rim_z"], tray_fit=receiver)
    return estimate