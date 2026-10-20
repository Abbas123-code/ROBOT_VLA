"""Subscribe to domain-44 wrist images; save a fresh, stable bottle estimate.

No motion commands and no simulator object-pose lookup. Source ROS Jazzy first.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["ROS_DOMAIN_ID"] = "44"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"


def capture(detector="color"):
    import numpy as np
    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from scipy.spatial.transform import Rotation
    from tf2_ros import Buffer, TransformListener
    from perception.bottle_rgbd import locate_bottle

    if detector == "official-yolov9":
        from perception.yolov9_official import OfficialYOLOv9
        model = OfficialYOLOv9()
        locate_bottle = model.locate
    elif detector != "color":
        raise ValueError("Unsupported detector")
    output_stem = "bottle_yolov9_detection" if detector == "official-yolov9" else "bottle_detection"

    rclpy.init(args=[])
    node = rclpy.create_node("bottle_rgbd_detector")
    buffer = Buffer(cache_time=Duration(seconds=10))
    listener = TransformListener(buffer, node)
    frames = {}
    infos = {}
    subs = []

    def receive(kind, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        frames.setdefault(stamp, {})[kind] = msg
        while len(frames) > 20:
            frames.pop(next(iter(frames)))

    for kind, topic in (("rgb", "/wrist_camera/color/image_raw"), ("depth", "/wrist_camera/depth/image_raw")):
        subs.append(node.create_subscription(Image, topic, lambda msg, key=kind: receive(key, msg), qos_profile_sensor_data))
    for key in ("color", "depth"):
        subs.append(node.create_subscription(CameraInfo, f"/wrist_camera/{key}/camera_info",
                    lambda msg, name=key: infos.__setitem__(name, msg), qos_profile_sensor_data))
    deadline = time.monotonic() + (60 if detector == "official-yolov9" else 35)
    estimates = []
    seen = set()
    diagnostic_saved = False
    last_error = "No matched RGB/depth/calibration/TF yet"
    try:
        while time.monotonic() < deadline and len(estimates) < 5:
            rclpy.spin_once(node, timeout_sec=.1)
            if set(infos) != {"color", "depth"}:
                continue
            for stamp, pair in list(frames.items()):
                if stamp in seen or set(pair) != {"rgb", "depth"}:
                    continue
                try:
                    rgb, depth = pair["rgb"], pair["depth"]
                    age = time.time() - stamp[0] - stamp[1] * 1e-9
                    if not 0 <= age < 5:
                        continue
                    if rgb.encoding != "rgb8" or depth.encoding != "32FC1" or rgb.is_bigendian or depth.is_bigendian:
                        raise ValueError("Unsupported image encoding")
                    info = infos["color"]
                    if any(msg.header.frame_id != info.header.frame_id for msg in (rgb, depth, infos["depth"])):
                        raise ValueError("Optical frames differ")
                    if list(info.k) != list(infos["depth"].k):
                        raise ValueError("Depth is not registered")
                    if any((m.width, m.height) != (info.width, info.height) for m in (rgb, depth)):
                        raise ValueError("Camera resolutions differ")
                    t = buffer.lookup_transform("world", info.header.frame_id, Time.from_msg(rgb.header.stamp))
                    # Do not repeatedly process a rejected frame and starve TF callbacks.
                    seen.add(stamp)
                    matrix = np.eye(4)
                    q, p = t.transform.rotation, t.transform.translation
                    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    matrix[:3, 3] = [p.x, p.y, p.z]
                    color = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.step)[:, :rgb.width*3].reshape(rgb.height, rgb.width, 3)
                    z = np.frombuffer(depth.data, dtype="<f4").reshape(depth.height, depth.step // 4)[:, :depth.width]
                    if not diagnostic_saved:
                        (ROOT / "ros2_ws/log").mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(ROOT / "ros2_ws/log/bottle_rgbd_input.npz",
                                            rgb=color, depth=z, k=np.array(info.k), camera_to_world=matrix)
                        diagnostic_saved = True
                    estimate = locate_bottle(color, z, info.k, matrix)
                    estimate["image_stamp"] = list(stamp)
                    estimates.append(estimate)
                    seen.add(stamp)
                    print("BOTTLE_RGBD_FRAME " + json.dumps(estimate), flush=True)
                    if len(estimates) == 5:
                        from PIL import Image as PillowImage, ImageDraw
                        annotated = PillowImage.fromarray(color)
                        draw = ImageDraw.Draw(annotated)
                        draw.rectangle(estimate["bbox_xyxy"], outline="red", width=2)
                        draw.text((10, 10), f"Bottle: {detector} + RGB-D fit", fill="red")
                        (ROOT / "ros2_ws/log").mkdir(parents=True, exist_ok=True)
                        annotated.save(ROOT / f"ros2_ws/log/{output_stem}.png")
                        break
                except Exception as error:
                    last_error = str(error)
                    if stamp in seen:
                        print("BOTTLE_FRAME_REJECTED: " + last_error, flush=True)
        if len(estimates) < 5:
            raise RuntimeError(f"Bottle detection unavailable: {last_error}")
        centres = np.array([item["center_xy"] for item in estimates])
        center = np.median(centres, axis=0)
        spread = float(np.max(np.linalg.norm(centres - center, axis=1)))
        if spread > .003:
            raise RuntimeError(f"Unstable bottle estimate: {spread} m")
        report = {"completed": True, "captured_at_unix": time.time(), "center_xy": center.tolist(),
                  "max_spread_m": spread, "frames": estimates, "source": "ROS RGB + depth + CameraInfo + TF",
                  "world_table_z": 0., "known_body_radius_m": .03225, "uses_simulator_object_pose": False,
                  "detector": detector}
        path = ROOT / f"ros2_ws/log/{output_stem}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("BOTTLE_DETECTION_PASSED " + json.dumps({"center_xy": center.tolist(), "spread_m": spread}), flush=True)
        return report
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("color", "official-yolov9"), default="color")
    capture(parser.parse_args().detector)