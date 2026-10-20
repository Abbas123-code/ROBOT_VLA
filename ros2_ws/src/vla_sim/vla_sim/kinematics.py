import math
import xml.etree.ElementTree as ET

import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if not math.isfinite(angle) or not np.isfinite(length) or length == 0:
        raise ValueError("Invalid rotation axis or angle")
    x, y, z = axis / length
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def link_transform(urdf_path, positions, target="panda_grasptarget", base="panda_link0"):
    """Return base-from-target using URDF origins and measured joint positions."""
    robot = ET.parse(urdf_path).getroot()
    joints = {joint.find("child").attrib["link"]: joint for joint in robot.findall("joint")}
    chain, visited = [], set()
    link = target
    while link != base:
        if link in visited or link not in joints:
            raise ValueError(f"No valid chain from {base} to {target}")
        visited.add(link)
        joint = joints[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        attributes = origin.attrib if origin is not None else {}
        xyz = np.array([float(v) for v in attributes.get("xyz", "0 0 0").split()])
        roll, pitch, yaw = [float(v) for v in attributes.get("rpy", "0 0 0").split()]
        local = np.eye(4)
        local[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
        local[:3, 3] = xyz
        motion = np.eye(4)
        kind = joint.attrib["type"]
        if kind != "fixed":
            value = float(positions[joint.attrib["name"]])
            if not math.isfinite(value):
                raise ValueError("Non-finite joint position")
            axis_element = joint.find("axis")
            axis = np.array([float(v) for v in (
                axis_element.attrib.get("xyz", "1 0 0") if axis_element is not None else "1 0 0"
            ).split()])
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = rotation(axis, value)
            elif kind == "prismatic":
                norm = np.linalg.norm(axis)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("Invalid translation axis")
                motion[:3, 3] = axis / norm * value
            else:
                raise ValueError(f"Unsupported joint type: {kind}")
        transform = transform @ local @ motion
    if not np.isfinite(transform).all():
        raise ValueError("Non-finite transform")
    return transform