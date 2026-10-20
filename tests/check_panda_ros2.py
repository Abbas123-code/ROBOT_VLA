import os
from pathlib import Path
import subprocess
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

from vla_sim.joint_commander import JointCommander
from vla_sim.motion import ordered_positions


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["ROS_DOMAIN_ID"] = "42"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    log_path = root / "ros2_ws/log/panda_integration.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=["--ros-args", "-r", "__ns:=/vla_sim",
                     "-p", "use_sim_time:=true", "-p", "enable_motion:=true"])
    commander = JointCommander()
    probe = Node("integration_probe")
    states, commands, clocks, invalid = [], [], [], []

    def receive(message):
        try:
            states.append(ordered_positions(message.name, message.position))
        except (ValueError, KeyError) as error:
            invalid.append(str(error))

    probe.create_subscription(JointState, "joint_states", receive, qos_profile_sensor_data)
    probe.create_subscription(JointState, "joint_command", commands.append, 10)
    probe.create_subscription(Clock, "/clock", clocks.append, qos_profile_sensor_data)
    executor = SingleThreadedExecutor()
    executor.add_node(commander)
    executor.add_node(probe)
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen([
                str(root / "env_isaacsim/bin/python"), str(root / "isaac/scripts/panda_tabletop.py"),
                "--headless", "--steps", "1200",
            ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while process.poll() is None and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            if process.poll() is None:
                raise TimeoutError(f"Simulation did not finish; see {log_path}")
        assert process.returncode == 0, f"Simulation failed; see {log_path}"
        assert not invalid, f"Invalid state messages: {invalid[:3]}"
        assert len(states) > 100, f"Only {len(states)} joint states; see {log_path}"
        assert len(commands) > 50, f"Only {len(commands)} joint commands"
        assert len(clocks) > 100, f"Only {len(clocks)} clock messages"
        movement = max(q[0] for q in states) - min(q[0] for q in states)
        assert movement > 0.05, f"Robot did not follow commands: joint1 range={movement}"
        print(f"PASS: {len(states)} states, {len(commands)} commands, {len(clocks)} clocks; "
              f"joint1 moved {movement:.3f} rad")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        executor.shutdown()
        commander.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()