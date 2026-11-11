"""Verified local fine-tuned weights using the unmodified official dual model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class TabletopYOLOv9:
    def __init__(self):
        self.config = json.loads((ROOT / "configs/yolov9_tabletop.json").read_text())
        repo, weight = ROOT / "third_party/yolov9", ROOT / self.config["weights"]
        rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
        if rev != self.config["revision"] or dirty:
            raise RuntimeError("Expected pinned unmodified WongKinYiu/yolov9")
        if hashlib.sha256(weight.read_bytes()).hexdigest() != self.config["weights_sha256"]:
            raise RuntimeError("Trained model checksum mismatch")
        if importlib.util.find_spec("ultralytics") is not None:
            raise RuntimeError("Use the separate environment without Ultralytics")
        os.environ["YOLOv5_AUTOINSTALL"] = "false"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(repo))
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        self.torch, self.letterbox, self.nms, self.scale_boxes = torch, letterbox, non_max_suppression, scale_boxes
        torch.set_num_threads(4)
        checkpoint = torch.load(str(weight), map_location="cpu", weights_only=False)
        self.model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        self.names = self.model.names
        actual = [self.names[i] for i in range(len(self.names))]
        if actual != self.config["names"] or self.model.model[-1].__class__.__name__ != "DualDDetect":
            raise RuntimeError("Checkpoint class mapping or head differs from trained model")
        self.stride = int(self.model.stride.max())

    def detect(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB image")
        padded = self.letterbox(rgb, self.config["image_size"], stride=self.stride, auto=True)[0]
        tensor = self.torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float()[None] / 255
        with self.torch.inference_mode():
            # Official detect_dual.py uses prediction[0][1]: the main inference head.
            prediction = self.model(tensor)[0][1]
            detections = self.nms(prediction, self.config["confidence_threshold"], self.config["iou_threshold"], max_det=30)[0]
        detections = detections.clone()
        result = []
        if len(detections):
            detections[:, :4] = self.scale_boxes(tensor.shape[2:], detections[:, :4], rgb.shape).round()
            for *box, confidence, cid in detections.tolist():
                result.append({"class_id": int(cid), "label": self.names[int(cid)],
                               "confidence": confidence, "bbox_xyxy": [int(v) for v in box]})
        return result