"""Read-only ROS subscriber check; run with camera publisher already open (domain 44)."""
import json
import os
from pathlib import Path
import time

os.environ["ROS_DOMAIN_ID"] = "44"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"


def main():
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_msgs.msg import TFMessage

    rclpy.init(args=[])
    node = rclpy.create_node("check_wrist_camera")
    received = {}
    stamps = {}
    subscriptions = []

    def collect(name, msg):
        received[name] = msg
        if name in ("rgb", "depth"):
            stamps.setdefault(name, set()).add((msg.header.stamp.sec, msg.header.stamp.nanosec))

    for name, topic, cls in (
        ("rgb", "/wrist_camera/color/image_raw", Image),
        ("depth", "/wrist_camera/depth/image_raw", Image),
        ("info", "/wrist_camera/color/camera_info", CameraInfo),
        ("depth_info", "/wrist_camera/depth/camera_info", CameraInfo),
        ("tf", "/tf", TFMessage),
    ):
        subscriptions.append(node.create_subscription(cls, topic, lambda msg, key=name: collect(key, msg), qos_profile_sensor_data))
    deadline = time.monotonic() + 40
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if len(received) == 5 and all(len(stamps.get(k, ())) >= 3 for k in ("rgb", "depth")):
                break
        assert len(received) == 5, f"Missing topics: {received.keys()}"
        assert all(len(stamps.get(k, ())) >= 3 for k in ("rgb", "depth")), "No fresh frames"
        color, depth = received["rgb"], received["depth"]
        assert color.encoding == "rgb8" and depth.encoding == "32FC1"
        assert (color.width, color.height, color.step) == (640, 480, 1920)
        assert (depth.width, depth.height, depth.step) == (640, 480, 2560)
        assert len(color.data) == 640 * 480 * 3 and len(depth.data) == 640 * 480 * 4
        assert not color.is_bigendian and not depth.is_bigendian
        z = np.frombuffer(depth.data, dtype="<f4")
        assert np.isfinite(z).mean() > 0.1 and np.nanmin(z) >= 0.02
        assert np.frombuffer(color.data, dtype=np.uint8).std() > 5
        for name in ("rgb", "depth", "info", "depth_info"):
            assert received[name].header.frame_id == "wrist_camera_optical_frame"
        assert list(received["info"].k) == [320., 0., 320., 0., 320., 240., 0., 0., 1.]
        assert list(received["depth_info"].k) == list(received["info"].k)
        assert stamps["rgb"] & stamps["depth"], "RGB/depth stamps are not synchronized"
        edges = {(t.header.frame_id, t.child_frame_id) for t in received["tf"].transforms}
        assert edges == {("world", "panda_hand"), ("panda_hand", "wrist_camera_optical_frame")}
        report = {"completed": True, "domain": 44, "received": sorted(received),
                  "fresh_rgb_frames": len(stamps["rgb"]), "fresh_depth_frames": len(stamps["depth"]),
                  "valid_depth_fraction": float(np.isfinite(z).mean()), "robot_commands_sent": False}
        path = Path(__file__).resolve().parents[1] / "ros2_ws/log/wrist_camera_ros_validation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("WRIST_CAMERA_ROS_TEST_PASSED", json.dumps(report), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()