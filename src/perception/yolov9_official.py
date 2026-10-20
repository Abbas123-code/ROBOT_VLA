"""CPU inference using only the pinned WongKinYiu/yolov9 implementation.

No ultralytics import or torch.hub execution; no upstream auto-install/download.
The official checkpoint pickles model classes. Full deserialization is allowed
ONLY after checking its pinned SHA256 and the clean, pinned upstream checkout.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class OfficialYOLOv9:
    def __init__(self):
        self.config = json.loads((ROOT / "configs/yolov9_official.json").read_text())
        repo = ROOT / "third_party/yolov9"
        weight = ROOT / "data/checkpoints/yolov9-s-converted.pt"
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
        if revision != self.config["revision"] or dirty:
            raise RuntimeError("Expected pinned, unmodified official YOLOv9 checkout")
        if hashlib.sha256(weight.read_bytes()).hexdigest() != self.config["weights_sha256"]:
            raise RuntimeError("Official YOLOv9-S checkpoint checksum mismatch")
        os.environ["YOLOv5_AUTOINSTALL"] = "false"  # upstream environment variable name
        sys.path.insert(0, str(repo))
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        self.torch = torch
        self.letterbox = letterbox
        self.nms = non_max_suppression
        self.scale_boxes = scale_boxes
        torch.set_num_threads(4)
        # Explicit modern-PyTorch compatibility; no change to global torch.load.
        checkpoint = torch.load(str(weight), map_location="cpu", weights_only=False)
        self.model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        self.names = self.model.names
        self.stride = int(self.model.stride.max())
        self.confidence = float(self.config["confidence_threshold"])

    def _detect_view(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB image")
        # Input already RGB: do NOT perform detect.py's BGR->RGB reversal.
        padded = self.letterbox(rgb, self.config["image_size"], stride=self.stride, auto=True)[0]
        tensor = self.torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float()[None] / 255
        with self.torch.inference_mode():
            prediction = self.model(tensor)
            detections = self.nms(prediction, self.confidence, self.config["iou_threshold"], max_det=30)[0]
        detections = detections.clone()
        result = []
        if len(detections):
            detections[:, :4] = self.scale_boxes(tensor.shape[2:], detections[:, :4], rgb.shape).round()
            for *box, confidence, class_id in detections.tolist():
                result.append({"class_id": int(class_id), "label": self.names[int(class_id)],
                               "confidence": confidence, "bbox_xyxy": [int(v) for v in box]})
        return result

    def detect(self, rgb):
        """Full image plus uniform overlapping tiles; no target-specific crop."""
        import torchvision
        objects = self._detect_view(rgb)
        height, width = rgb.shape[:2]
        size = int(self.config["tile_size"])
        step = int(size * (1 - self.config["tile_overlap"]))
        if height > size or width > size:
            ys = sorted(set([*range(0, max(1, height-size+1), step), max(0, height-size)]))
            xs = sorted(set([*range(0, max(1, width-size+1), step), max(0, width-size)]))
            for y in ys:
                for x in xs:
                    for item in self._detect_view(rgb[y:y+size, x:x+size]):
                        a, b, c, d = item["bbox_xyxy"]
                        item["bbox_xyxy"] = [a+x, b+y, c+x, d+y]
                        objects.append(item)
        if not objects:
            return []
        boxes = self.torch.tensor([item["bbox_xyxy"] for item in objects], dtype=self.torch.float32)
        scores = self.torch.tensor([item["confidence"] for item in objects])
        classes = self.torch.tensor([item["class_id"] for item in objects])
        keep = torchvision.ops.batched_nms(boxes, scores, classes, self.config["iou_threshold"])
        return [objects[index] for index in keep.tolist()]

    def locate(self, rgb, depth, k, camera_to_world):
        from perception.bottle_rgbd import locate_bottle
        objects = self.detect(rgb)
        bottles = [item for item in objects if item["label"] == "bottle"]
        if len(bottles) != 1:
            raise ValueError(f"Official YOLOv9-S requires one bottle >= {self.confidence}: {objects}")
        bottle = bottles[0]
        x1, y1, x2, y2 = bottle["bbox_xyxy"]
        # Depth/known-radius fitting refines only the independently recognized
        # bottle ROI. No colour-only fallback when YOLO does not find a bottle.
        roi = np.zeros_like(rgb)
        roi[max(0, y1):y2, max(0, x1):x2] = rgb[max(0, y1):y2, max(0, x1):x2]
        estimate = locate_bottle(roi, depth, k, camera_to_world)
        estimate.update(detector="WongKinYiu/yolov9", model="yolov9-s-converted.pt",
                        yolo_bottle=bottle, yolo_objects=objects,
                        weights_sha256=self.config["weights_sha256"], revision=self.config["revision"])
        return estimate