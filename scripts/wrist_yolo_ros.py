"""Publish trained YOLO bounding boxes from raw wrist RGB; optional fresh RGB-D capture.

No motor publishers. --capture requires five stable, exactly time-matched RGB,
depth, CameraInfo and TF observations of both bottle and tray before saving.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def stamp_key(msg):
    return (msg.header.stamp.sec, msg.header.stamp.nanosec)


def rgb_array(msg):
    import numpy as np
    if msg.encoding != "rgb8" or msg.is_bigendian or msg.step < msg.width*3:
        raise ValueError("Expected little-endian rgb8")
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3].reshape(msg.height, msg.width, 3).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--domain", type=int, default=44)
    parser.add_argument("--max-frames", type=int, default=0, help="Bounded publisher integration check")
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    import numpy as np
    from PIL import Image as PILImage, ImageDraw
    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    from perception.yolov9_tabletop import TabletopYOLOv9
    from perception.tabletop_rgbd import locate_targets

    model = TabletopYOLOv9()
    rclpy.init(args=[])
    node = rclpy.create_node("tabletop_target_capture" if args.capture else "wrist_tabletop_yolo")
    image_pub = node.create_publisher(Image, "/wrist_camera/detections/image", 1)
    objects_pub = node.create_publisher(String, "/wrist_camera/detections/objects", 1)
    buffer = Buffer(cache_time=Duration(seconds=15))
    listener = TransformListener(buffer, node)
    frames, subscriptions, seen = {}, [], deque(maxlen=100)
    estimates = deque(maxlen=5)
    count, last_error = 0, "No synchronized RGB-D data"
    deadline = time.monotonic()+60
    reports = ROOT / "ros2_ws/log"
    reports.mkdir(parents=True, exist_ok=True)

    def receive(kind, msg):
        frames.setdefault(stamp_key(msg), {})[kind] = msg
        while len(frames) > 20:
            del frames[next(iter(frames))]

    for kind, topic, msg_type in (("rgb", "color/image_raw", Image), ("depth", "depth/image_raw", Image),
                                  ("info", "color/camera_info", CameraInfo), ("depth_info", "depth/camera_info", CameraInfo)):
        subscriptions.append(node.create_subscription(msg_type, "/wrist_camera/"+topic,
                             lambda msg, key=kind: receive(key, msg), qos_profile_sensor_data))
    try:
        print(f"WRIST_YOLO_READY domain={args.domain} model={model.config['weights_sha256']} capture={args.capture}", flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.02)
            if args.capture and time.monotonic() > deadline:
                raise RuntimeError("Fresh stable bottle/tray capture failed: "+last_error)
            candidates = [(s, p) for s, p in frames.items() if s not in seen and
                          (set(p) == {"rgb", "depth", "info", "depth_info"} if args.capture else "rgb" in p)]
            if not candidates:
                continue
            stamp, pair = max(candidates)
            age = time.time()-stamp[0]-stamp[1]*1e-9
            if not 0 <= age < 2:
                seen.append(stamp)
                continue
            if args.capture:
                try:
                    transform = buffer.lookup_transform("world", pair["rgb"].header.frame_id, Time.from_msg(pair["rgb"].header.stamp))
                except Exception:
                    continue  # Allow TF callback to arrive; do not mark this frame seen yet.
            seen.append(stamp)
            rgb = rgb_array(pair["rgb"])
            started = time.monotonic()
            objects = model.detect(rgb)
            latency = time.monotonic()-started
            canvas = PILImage.fromarray(rgb)
            draw = ImageDraw.Draw(canvas)
            for obj in objects:
                color = "cyan" if obj["class_id"] == 0 else "lime" if obj["class_id"] == 6 else "yellow"
                draw.rectangle(obj["bbox_xyxy"], outline=color, width=2)
                x, y = obj["bbox_xyxy"][:2]
                draw.text((x, max(14, y)-12), f"{obj['label']} {obj['confidence']:.2f}", fill=color, stroke_width=1, stroke_fill="black")
            draw.text((4, 4), f"YOLOv9-S custom | {len(objects)} objects | {latency*1000:.0f} ms | NO MOTION COMMANDS", fill="white", stroke_width=1, stroke_fill="black")
            annotated = Image()
            annotated.header = pair["rgb"].header
            annotated.width, annotated.height = canvas.size
            annotated.encoding, annotated.is_bigendian = "rgb8", 0
            annotated.step, annotated.data = annotated.width*3, canvas.tobytes()
            # Capture is a separate consumer: it must not contend with the live annotated publisher.
            if not args.capture:
                image_pub.publish(annotated)
                objects_pub.publish(String(data=json.dumps({"image_stamp": stamp, "frame_id": annotated.header.frame_id,
                    "objects": objects, "inference_seconds": latency, "weights_sha256": model.config["weights_sha256"]})))
            count += 1
            if count == 1 or count % 20 == 0:
                canvas.save(reports / f"wrist_yolo_domain{args.domain}.png")
                print("WRIST_YOLO_FRAME "+json.dumps({"count": count, "image_stamp": stamp, "objects": objects}), flush=True)
            if args.capture:
                try:
                    info, dinfo, depth = pair["info"], pair["depth_info"], pair["depth"]
                    if any(m.header.frame_id != info.header.frame_id for m in pair.values()):
                        raise ValueError("Optical frame mismatch")
                    if list(info.k) != list(dinfo.k) or any((m.width, m.height) != (info.width, info.height) for m in pair.values()):
                        raise ValueError("RGB/depth registration mismatch")
                    if depth.encoding != "32FC1" or depth.is_bigendian or depth.step % 4:
                        raise ValueError("Unsupported depth encoding")
                    z = np.frombuffer(depth.data, dtype="<f4").reshape(depth.height, depth.step//4)[:, :depth.width]
                    q, p = transform.transform.rotation, transform.transform.translation
                    matrix = np.eye(4)
                    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    matrix[:3, 3] = [p.x, p.y, p.z]
                    estimate = locate_targets(rgb, z, info.k, matrix, objects)
                    estimate["image_stamp"] = list(stamp)
                    estimates.append(estimate)
                    if len(estimates) == 5:
                        if stamp[0]+stamp[1]*1e-9 - (estimates[0]["image_stamp"][0]+estimates[0]["image_stamp"][1]*1e-9) > 10:
                            raise ValueError("Observation window too old")
                        positions = np.array([[*e["center_xy"], *e["tray_center_xy"], e["tray_floor_z"]] for e in estimates])
                        median = np.median(positions, axis=0)
                        spread = np.max(np.linalg.norm(positions-median, axis=1))
                        if spread > .003:
                            raise ValueError("Targets unstable over five observations")
                        report = {"completed": True, "captured_at_unix": stamp[0]+stamp[1]*1e-9,
                                  "center_xy": median[:2].tolist(), "tray_center_xy": median[2:4].tolist(),
                                  "tray_floor_z": float(median[4]), "tray_rim_z": .06,
                                  "max_spread_m": float(spread), "frames": list(estimates),
                                  "detector": "trained-yolov9-tabletop", "weights_sha256": model.config["weights_sha256"],
                                  "source": "exact-stamp ROS RGB-D + CameraInfo + TF", "uses_simulator_object_pose": False}
                        (reports / "tabletop_detection.json").write_text(json.dumps(report, indent=2)+"\n")
                        canvas.save(reports / "tabletop_detection.png")
                        print("TABLETOP_CAPTURE_PASSED "+json.dumps({k: v for k, v in report.items() if k != "frames"}), flush=True)
                        return
                except ValueError as error:
                    last_error = str(error)
                    estimates.clear()
                    print("TABLETOP_FRAME_REJECTED "+last_error, flush=True)
            if args.max_frames and count >= args.max_frames:
                return
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()