"""Read-only domain-46 verification: moving joints and matching raw/YOLO video."""
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    os.environ["ROS_DOMAIN_ID"] = "46"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    rclpy.init(args=[])
    node = rclpy.create_node("verify_visible_motion_readonly")
    raw, annotated, joints, sources = {}, set(), [], []

    def image(message, overlay=False):
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        age = time.time() - stamp[0] - stamp[1]*1e-9
        if not 0 <= age <= 3 or message.encoding != "rgb8":
            return
        if overlay:
            annotated.add(stamp)
        else:
            raw[stamp] = hashlib.sha256(bytes(message.data)).hexdigest()

    subscriptions = [
        node.create_subscription(Image, "/wrist_camera/color/image_raw", image, qos_profile_sensor_data),
        node.create_subscription(Image, "/wrist_camera/detections/image", lambda m: image(m, True), qos_profile_sensor_data),
        node.create_subscription(JointState, "/vla_pick_place/joint_states", lambda m: joints.append(list(m.position)), 10),
        node.create_subscription(String, "/wrist_camera/source", lambda m: sources.append(json.loads(m.data)), 10),
    ]
    deadline = time.monotonic() + 30
    passed = False
    delta = 0.
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if len(joints) > 1:
                delta = float(np.max(np.abs(np.array(joints[-1])-np.array(joints[0]))))
            passed = (len(set(raw.values())) >= 3 and len(annotated.intersection(raw)) >= 3
                      and delta > .001 and len(sources) >= 3
                      and all(s["domain"] == 46 and s["source"] == "motion_validation" for s in sources))
            if passed:
                break
        report = {"completed": passed, "domain": 46, "distinct_raw_images": len(set(raw.values())),
                  "matching_raw_annotated_stamps": len(annotated.intersection(raw)),
                  "max_joint_change_rad_or_m": delta, "source_samples": len(sources),
                  "scope": "Read-only video/joint motion check, NOT grasp or placement approval"}
        output = Path(__file__).resolve().parents[1] / "ros2_ws/log/live_motion_view_validation.json"
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError("Moving joint and same-stamp image evidence not observed within 30 seconds")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()