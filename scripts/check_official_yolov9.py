"""Read-only inference on a recorded wrist RGB image. Never commands a robot."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perception.yolov9_official import OfficialYOLOv9


def main():
    frame = np.load(ROOT / "ros2_ws/log/bottle_rgbd_input.npz")
    detector = OfficialYOLOv9()
    objects = detector.detect(frame["rgb"])
    image = Image.fromarray(frame["rgb"])
    draw = ImageDraw.Draw(image)
    for item in objects:
        draw.rectangle(item["bbox_xyxy"], outline="red", width=2)
        x, y = item["bbox_xyxy"][:2]
        draw.text((x, max(0, y-12)), f'{item["label"]} {item["confidence"]:.2f}', fill="red")
    image.save(ROOT / "ros2_ws/log/yolov9_official_detection.png")
    report = {"source": "recorded RGB, not live", "objects": objects, "config": detector.config,
              "motion_authorized_by_this_test": False}
    try:
        report["localization"] = detector.locate(frame["rgb"], frame["depth"], frame["k"], frame["camera_to_world"])
        report["bottle_accepted"] = True
    except ValueError as error:
        report["bottle_accepted"] = False
        report["reason"] = str(error)
    (ROOT / "ros2_ws/log/yolov9_official_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()