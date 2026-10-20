"""Run inside the open Isaac Sim Script Editor; reads poses without editing the scene."""

import json

import carb
import numpy as np
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.xforms import get_world_pose
from pxr import Gf, Usd, UsdGeom


def inspect_positions():
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Open the Panda tabletop scene first")
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_stopped():
        raise RuntimeError("Press Play first; Pause is also okay after simulation has started")
    fabric_stage = usdrt.Usd.Stage.Attach(context.get_stage_id())

    def unique_link(name):
        paths = [str(prim.GetPath()) for prim in Usd.PrimRange(
            stage.GetPrimAtPath("/World/panda"), Usd.TraverseInstanceProxies()
        ) if prim.GetName() == name and prim.IsA(UsdGeom.Xform)]
        if len(paths) != 1:
            raise RuntimeError(f"Expected exactly one {name}, found {paths}")
        return paths[0]

    def pose(path):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Missing prim: {path}")
        prim = fabric_stage.GetPrimAtPath(path)
        has_fabric_pose = bool(prim) and usdrt.Rt.Xformable(prim).HasWorldXform()
        if not has_fabric_pose and not carb.settings.get_settings().get("/physics/updateToUsd"):
            raise RuntimeError(f"No verified live pose for {path}; refusing to report a saved pose as live")
        position, quaternion = get_world_pose(path, fabric=has_fabric_pose)
        return np.asarray(position), np.asarray(quaternion)

    base, base_quaternion = pose(unique_link("panda_link0"))
    hand, quaternion = pose(unique_link("panda_hand"))
    cube, _ = pose("/World/Cube")
    rotation = Gf.Rotation(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:].tolist())))
    # This Panda URDF places panda_grasptarget 0.105 m along the hand's local Z axis.
    grasp = hand + np.array(rotation.TransformDir(Gf.Vec3d(0, 0, 0.105)))
    units = UsdGeom.GetStageMetersPerUnit(stage)
    base, hand, cube, grasp = (value * units for value in (base, hand, cube, grasp))
    base_rotation = Gf.Rotation(Gf.Quatd(
        float(base_quaternion[0]), Gf.Vec3d(*base_quaternion[1:].tolist())
    ))
    # This scene uses a 5 cm cube; 0.125 m above its centre clears its top by 10 cm when upright.
    approach_world = cube + np.array([0.0, 0.0, 0.125])
    approach_base = np.array(base_rotation.GetInverse().TransformDir(
        Gf.Vec3d(*(approach_world - base).tolist())
    ))
    report = {
        "source": "currently open Isaac Sim scene, runtime poses",
        "frame": "world", "units": "metres", "timeline_time": timeline.get_current_time(),
        "base_xyz": base.tolist(), "hand_origin_xyz": hand.tolist(),
        "base_quaternion_wxyz": base_quaternion.tolist(),
        "hand_quaternion_wxyz": quaternion.tolist(),
        "grasp_target_xyz": grasp.tolist(), "cube_center_xyz": cube.tolist(),
        "approach_target_world_xyz": approach_world.tolist(),
        "approach_target_base_xyz": approach_base.tolist(),
        "cube_minus_grasp_xyz": (cube - grasp).tolist(),
        "distance_metres": float(np.linalg.norm(cube - grasp)),
        "note": "Distance alone does not prove IK reachability or a collision-free approach.",
    }
    print(json.dumps(report, indent=2))
    return report


inspect_positions()