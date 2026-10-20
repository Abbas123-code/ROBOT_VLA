import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from vla_sim.motion import JOINT_NAMES, demo_target, ordered_positions


class JointCommander(Node):
    def __init__(self):
        super().__init__("joint_commander")
        self.declare_parameter("enable_motion", False)
        self.publisher = self.create_publisher(JointState, "joint_command", 10)
        self.subscription = self.create_subscription(
            JointState, "joint_states", self.receive_state, qos_profile_sensor_data
        )
        self.initial = None
        self.start_time = None
        self.last_received = 0.0
        self.timer = self.create_timer(0.05, self.publish_command)
        self.get_logger().info("Simulation commander ready; motion requires enable_motion:=true")

    def receive_state(self, message):
        try:
            positions = ordered_positions(message.name, message.position)
        except (ValueError, KeyError):
            self.get_logger().warning("Ignoring invalid or incomplete Panda joint state", throttle_duration_sec=5)
            return
        self.last_received = time.monotonic()
        if self.initial is None:
            self.initial = positions

    def publish_command(self):
        if not self.get_parameter("enable_motion").value:
            self.start_time = None
            self.initial = None
            return
        if self.initial is None or time.monotonic() - self.last_received > 1.0:
            self.start_time = None
            self.initial = None
            return
        now = self.get_clock().now()
        if self.start_time is None or now.nanoseconds < self.start_time:
            self.start_time = now.nanoseconds
        message = JointState()
        message.header.stamp = now.to_msg()
        message.name = list(JOINT_NAMES)
        message.position = demo_target(self.initial, (now.nanoseconds - self.start_time) / 1e9)
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = JointCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()