import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from vla_sim.kinematics import link_transform
from vla_sim.motion import JOINT_NAMES, ordered_positions


def main(args=None):
    parser = argparse.ArgumentParser(description="Read-only Panda FK from a fresh ROS joint-state message")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    options = parser.parse_args(args)
    if not options.urdf.is_file() or options.timeout <= 0:
        parser.error("Provide an existing URDF and a positive timeout")
    rclpy.init(args=[])
    node = rclpy.create_node("position_report", namespace="vla_sim")
    messages = []
    subscription = node.create_subscription(
        JointState, "joint_states", messages.append, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + options.timeout
        while not messages and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not messages:
            raise TimeoutError("No fresh joint states. Check Play and ROS_DOMAIN_ID=42; no commands were sent.")
        message = messages[-1]
        positions = dict(zip(JOINT_NAMES, ordered_positions(message.name, message.position)))
        hand = link_transform(options.urdf, positions, target="panda_hand")
        grasp = link_transform(options.urdf, positions)
        print(json.dumps({
            "source": "URDF forward kinematics from received joint feedback; not a direct simulator pose",
            "frame": "panda_link0", "units": "metres",
            "simulation_time": message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
            "hand_origin_xyz": hand[:3, 3].round(6).tolist(),
            "grasp_target_xyz": grasp[:3, 3].round(6).tolist(),
            "grasp_target_rotation_matrix": grasp[:3, :3].round(6).tolist(),
            "note": "World comparison requires the base pose. Cube pose is not published by the current graph.",
        }, indent=2))
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()