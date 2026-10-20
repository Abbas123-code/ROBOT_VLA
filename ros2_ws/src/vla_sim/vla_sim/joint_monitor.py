import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from vla_sim.motion import ordered_positions


class JointMonitor(Node):
    def __init__(self):
        super().__init__("joint_monitor")
        self.latest = None
        self.last_received = 0.0
        self.count = 0
        self.subscription = self.create_subscription(
            JointState, "joint_states", self.receive_state, qos_profile_sensor_data
        )
        self.timer = self.create_timer(2.0, self.report, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive_state(self, message):
        try:
            self.latest = ordered_positions(message.name, message.position)
        except (ValueError, KeyError):
            self.get_logger().warning("Invalid or incomplete Panda joint state", throttle_duration_sec=5)
            return
        self.last_received = time.monotonic()
        self.count += 1

    def report(self):
        if self.latest is None or time.monotonic() - self.last_received > 2.0:
            self.get_logger().warning("No fresh joint states. Check Play, ROS bridge, and ROS_DOMAIN_ID=42.")
            return
        values = ", ".join(f"{value:.3f}" for value in self.latest)
        self.get_logger().info(f"Received {self.count} states; arm positions [rad]: {values}")


def main(args=None):
    rclpy.init(args=args)
    node = JointMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()