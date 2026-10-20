"""URDF-based orientation-aware bottle path planning (no simulation commands).

Unlike the legacy cube IK, all seven joints may move. Small seed regularization
selects a nearby redundant solution; pose error and branch bounds remain strict.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


class BottlePosePlanner:
    def __init__(self, urdf, names, limits):
        self.limits = np.asarray(limits, dtype=float)
        root = ET.parse(urdf).getroot()
        joints = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
        chain = []
        link = "panda_grasptarget"
        while link != "panda_link0":
            joint = joints[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.chain = []
        for joint in reversed(chain):
            origin = joint.find("origin")
            attrs = origin.attrib if origin is not None else {}
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            kind = joint.attrib["type"]
            index, axis = None, None
            if kind != "fixed":
                if kind != "revolute":
                    raise ValueError("Only revolute arm joints are supported")
                index = names.index(joint.attrib["name"])
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                axis /= np.linalg.norm(axis)
            self.chain.append((transform, index, axis))

    def fk(self, angles):
        result = np.eye(4)
        for origin, index, axis in self.chain:
            result = result @ origin
            if index is not None:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * angles[index]).as_matrix()
                result = result @ motion
        return result

    def ik(self, position, orientation, seed):
        def residual(angles):
            frame = self.fk(angles)
            error = Rotation.from_matrix(orientation.T @ frame[:3, :3]).as_rotvec()
            return np.r_[frame[:3, 3] - position, .2 * error, 1e-4 * (angles - seed)]

        lo, hi = self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5
        result = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi),
                               max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not result.success or np.linalg.norm(result.fun[:6]) > .0001:
            raise RuntimeError(f"Pose IK failed at {position}: error={np.linalg.norm(result.fun[:6])}")
        return result.x

    def segment(self, seed, position, orientation, count=120):
        frame = self.fk(seed)
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([frame[:3, :3], orientation])))
        samples = [np.asarray(seed).copy()]
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            point = frame[:3, 3] + fraction * (np.asarray(position) - frame[:3, 3])
            samples.append(self.ik(point, rotations(fraction).as_matrix(), samples[-1]))
        samples = np.asarray(samples)
        if np.max(np.abs(np.diff(samples, axis=0))) > .12:
            raise RuntimeError("IK branch discontinuity")
        return samples


def body_orientation(target_xy, tilt_degrees=60, roll_sign=1):
    radial = np.r_[target_xy / np.linalg.norm(target_xy), 0.]
    theta = np.radians(tilt_degrees)
    forward = radial * np.sin(theta) - np.array([0., 0., np.cos(theta)])
    closing = roll_sign * np.cross([0., 0., 1.], radial)
    return np.column_stack((np.cross(closing, forward), closing, forward))