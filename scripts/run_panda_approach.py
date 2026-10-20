"""One-shot ROS open-finger motion for the validated Isaac scene; never grasps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/vla_sim"))

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

from vla_sim.approach import check_grasp_position, load_plan, sample
from vla_sim.kinematics import link_transform
from vla_sim.motion import JOINT_NAMES, ordered_positions


class Approach(Node):
    def __init__(self):
        super().__init__("hover_approach", namespace="/vla_sim")
        self.positions = self.stamp = self.sim_time = None
        self.received = self.clock_received = 0.0
        self.feedback_error = None
        self.fingers = None
        self.publisher = None
        self.create_subscription(JointState, "joint_states", self.on_state, qos_profile_sensor_data)
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
        self.client = self.create_client(GetParameters, "/vla_sim/joint_commander/get_parameters")
        self.pending = None
        self.checked = 0.0

    def on_state(self, message):
        try:
            positions = ordered_positions(message.name, message.position)
            mapping = dict(zip(message.name, message.position))
            fingers = [mapping[f"panda_finger_joint{i}"] for i in (1, 2)]
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if self.stamp is not None and stamp < self.stamp - 1e-6:
                raise ValueError("Joint feedback time went backwards")
            self.positions, self.fingers, self.stamp = positions, fingers, stamp
            self.received = time.monotonic()
        except (ValueError, KeyError) as error:
            self.feedback_error = str(error)

    def on_clock(self, message):
        value = message.clock.sec + message.clock.nanosec * 1e-9
        if self.sim_time is not None and value < self.sim_time - 1e-6:
            self.feedback_error = "Simulation clock went backwards"
        self.sim_time = value
        self.clock_received = time.monotonic()

    def check_demo(self):
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            if result is None or len(result.values) != 1:
                raise RuntimeError("Cannot read demo state")
            value = result.values[0]
            if value.type != ParameterType.PARAMETER_BOOL or value.bool_value:
                raise RuntimeError("Demo must remain disabled")
            self.checked, self.pending = time.monotonic(), None
        if self.pending is None and time.monotonic() - self.checked > 0.5:
            self.pending = self.client.call_async(GetParameters.Request(names=["enable_motion"]))

    def check_graph(self):
        for topic in ("/vla_sim/joint_states", "/clock"):
            if self.count_publishers(topic) != 1:
                raise RuntimeError(f"Expected exactly one simulation publisher on {topic}")
        endpoints = self.get_publishers_info_by_topic("/vla_sim/joint_command")
        allowed = {("/vla_sim", "joint_commander"), ("/vla_sim", "hover_approach")}
        identities = [(item.node_namespace, item.node_name) for item in endpoints]
        if any(identity not in allowed for identity in identities) or len(identities) != len(set(identities)):
            raise RuntimeError(f"Unexpected command publishers: {identities}")
        if self.count_subscribers("/vla_sim/joint_command") != 1:
            raise RuntimeError("Expected exactly one simulation command subscriber")

    def publish(self, positions):
        message = JointState()
        message.name = list(JOINT_NAMES)
        message.position = list(ordered_positions(JOINT_NAMES, positions))
        self.publisher.publish(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send validated simulation commands; default is read-only")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/panda_approach.json")
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "42":
        raise RuntimeError("This simulation-only script requires ROS_DOMAIN_ID=42")
    plan, digest = load_plan(args.plan)
    validation = json.loads((ROOT / "ros2_ws/log" / f"{args.plan.stem}_validation.json").read_text())
    execution_report = ROOT / "ros2_ws/log" / f"{args.plan.stem}_execution.json"
    urdf = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
    if not np.allclose(plan["base_quaternion_wxyz"], [1, 0, 0, 0], atol=1e-8):
        raise RuntimeError("This world-position estimate requires an identity-oriented base")

    def estimate_grasp(positions):
        transform = link_transform(urdf, dict(zip(JOINT_NAMES, positions)))
        estimate = np.asarray(plan["base_world_xyz"]) + transform[:3, 3]
        check_grasp_position(plan, estimate)
        return estimate

    scene_hash = hashlib.sha256((ROOT / "isaac/scenes/panda_tabletop.usda").read_bytes()).hexdigest()
    if not validation.get("passed") or validation.get("plan_sha256") != digest or validation.get("scene_sha256") != scene_hash:
        raise RuntimeError("A passing isolated test of this exact plan and saved scene is required")
    rclpy.init()
    node = Approach()
    commands = 0
    try:
        if not node.client.wait_for_service(timeout_sec=5):
            raise RuntimeError("Demo parameter service is unavailable")
        deadline = time.monotonic() + 10
        graph_ready_since = None
        graph_error = "ROS graph discovery did not complete"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            node.check_demo()
            try:
                node.check_graph()
                if graph_ready_since is None:
                    graph_ready_since = time.monotonic()
            except RuntimeError as error:
                graph_ready_since, graph_error = None, str(error)
            if (node.positions is not None and node.sim_time is not None and node.checked
                    and graph_ready_since is not None and time.monotonic() - graph_ready_since >= 1):
                break
        if graph_ready_since is None or time.monotonic() - graph_ready_since < 1:
            raise RuntimeError(graph_error)
        now = time.monotonic()
        if now - node.received > 0.5 or now - node.clock_received > 0.5 or not node.checked:
            raise RuntimeError("Fresh joint states, clock and disabled demo are required")
        if node.feedback_error:
            raise RuntimeError(node.feedback_error)
        node.check_graph()
        if np.max(np.abs(np.array(node.positions) - plan["start_arm_radians"])) > 0.015:
            raise RuntimeError(f"Start differs from validated configuration: {node.positions}")
        if not np.allclose(node.fingers, plan["finger_positions_metres"], atol=0.002):
            raise RuntimeError("Fingers are not open as validated")
        estimate_grasp(node.positions)
        print("PREFLIGHT_PASSED: fresh feedback, expected start, one simulator, demo disabled", flush=True)
        if not args.execute:
            return
        node.publisher = node.create_publisher(JointState, "joint_command", 10)
        start, last_clock = node.sim_time, node.sim_time
        clock_progress, last_publish = time.monotonic(), -float("inf")
        deadline = time.monotonic() + 90
        previous = list(node.positions)
        peak_error = 0.0
        finished = False
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            node.check_demo()
            if node.feedback_error:
                raise RuntimeError(node.feedback_error)
            if now - node.received > 0.5 or now - node.clock_received > 0.5 or now - node.checked > 2:
                raise RuntimeError("Feedback, clock, or demo status became stale")
            if node.sim_time > last_clock:
                if node.sim_time - last_clock > 0.25:
                    raise RuntimeError("Simulation clock jumped forward")
                clock_progress, last_clock = now, node.sim_time
            if now - clock_progress > 1:
                raise RuntimeError("Simulation stopped advancing")
            if not np.allclose(node.fingers, plan["finger_positions_metres"], atol=0.002):
                raise RuntimeError("Finger state changed unexpectedly")
            error = float(np.max(np.abs(np.array(node.positions) - previous)))
            peak_error = max(peak_error, error)
            if error > plan["max_joint_tracking_error_rad"]:
                raise RuntimeError(f"Joint tracking error {error:.4f} exceeds limit")
            estimate_grasp(node.positions)
            elapsed = node.sim_time - start
            if elapsed - last_publish >= 0.05:
                node.check_graph()
                previous = sample(plan, elapsed)
                node.publish(previous)
                commands += 1
                last_publish = elapsed
            if elapsed >= plan["duration_seconds"] + plan["settle_seconds"]:
                if np.max(np.abs(np.array(node.positions) - plan["target_arm_radians"])) > 0.02:
                    raise RuntimeError("Final joint error exceeds 0.02 rad")
                finished = True
                break
        if not finished:
            raise RuntimeError("Approach exceeded its wall-time deadline")
        estimate = estimate_grasp(node.positions)
        if np.linalg.norm(estimate - plan["target_world_xyz"]) > plan["max_final_position_error_metres"]:
            raise RuntimeError("Estimated final grasp position is outside the target tolerance")
        report = {"completed": True, "commands": commands, "plan_sha256": digest,
                  "final_arm_radians": node.positions, "peak_tracking_error_rad": peak_error,
                  "estimated_grasp_world_xyz": estimate.tolist(),
                  "estimated_target_error_metres": float(np.linalg.norm(estimate - plan["target_world_xyz"])),
                  "pose_note": "URDF FK plus previously measured identity-oriented base; not a new live world/cube measurement",
                  "fingers": node.fingers, "simulation_elapsed_seconds": elapsed}
        execution_report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        # A measured-position hold is best effort, not an emergency stop.
        if commands and node.positions is not None and time.monotonic() - node.received < 0.5:
            node.publish(node.positions)
            rclpy.spin_once(node, timeout_sec=0.05)
            print("ABORT: requested hold at last measured joint positions", flush=True)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()