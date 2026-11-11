"""Fine-tune the pinned official dual-head YOLOv9-S; never command robot motion."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/yolov9"
REVISION = "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff"
WEIGHTS = ROOT / "data/checkpoints/yolov9-s.pt"
WEIGHTS_SHA256 = "e232b9c0423ff04608d973e684f8ee808f933e6b7fc052787efb30e43b8acac5"


def enable_legacy_font_size():
    """Pillow>=10 removed getsize, still used by upstream annotation plots."""
    from PIL import ImageFont

    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def getsize(self, text):
            left, top, right, bottom = self.getbbox(text)
            return right - left, bottom
        setattr(ImageFont.FreeTypeFont, "getsize", getsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/datasets/yolov9_tabletop_v1/dataset.yaml")
    parser.add_argument("--name", default="tabletop_yolov9s_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.data = args.data.resolve()
    output = ROOT / "outputs/training" / args.name
    if Path(args.name).name != args.name or args.name in (".", "..") or output.exists():
        raise ValueError("Use a new single-component run name; existing runs are protected")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("Official source revision differs from pinned revision")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code", "HEAD", "--"], check=True)
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("Pretrained checkpoint checksum mismatch")
    if importlib.util.find_spec("ultralytics") is not None:
        raise RuntimeError("Use the isolated environment without the Ultralytics package")
    if not args.data.is_file():
        raise FileNotFoundError("Prepare the dataset before training")
    # Scoped to this trusted official/local training process, not global settings.
    # Upstream uses full-model pickle checkpoints and predates torch>=2.6 defaults.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UPSTREAM))
    enable_legacy_font_size()
    import torch
    import torchvision

    torch.set_num_threads(8)
    if args.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        device = torch.device(f"cuda:{args.device}")
        sample = torch.randn((32, 32), device=device, requires_grad=True)
        sample.square().sum().backward()
        torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device=device),
                            torch.tensor([0.9], device=device), 0.5)
        torch.cuda.synchronize(device)
    import train_dual

    os.chdir(UPSTREAM)
    sys.argv = [str(UPSTREAM / "train_dual.py"),
                "--weights", str(WEIGHTS), "--cfg", str(UPSTREAM / "models/detect/yolov9-s.yaml"),
                "--data", str(args.data.resolve()), "--hyp", str(ROOT / "configs/yolov9_tabletop_hyp.yaml"),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--imgsz", "640", "--device", args.device, "--workers", str(args.workers),
                "--optimizer", "AdamW", "--cos-lr", "--patience", "25", "--seed", "42",
                "--close-mosaic", str(min(10, args.epochs // 5)), "--project", str(output.parent),
                "--name", args.name, "--cache", "ram"]
    provenance = {"repository": "https://github.com/WongKinYiu/yolov9", "revision": revision,
                  "pretrained_url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt",
                  "pretrained_sha256": WEIGHTS_SHA256, "torch": torch.__version__,
                  "torchvision": torchvision.__version__, "command": sys.argv,
                  "dataset_yaml_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  "robot_motion": False}
    print(json.dumps(provenance, indent=2), flush=True)
    train_dual.main(train_dual.parse_opt())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best = output / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError("Training returned without a best checkpoint")
    with (output / "results.csv").open() as handle:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise RuntimeError("Missing or nonfinite training metrics")
    best_row = max(rows, key=lambda r: 0.1*r["metrics/mAP_0.5"] + 0.9*r["metrics/mAP_0.5:0.95"])
    summary = {"completed": True, "epochs_completed": len(rows), "best_epoch_metrics": best_row,
               "best_checkpoint": str(best), "best_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
               "warning": "Small single-scene validation only; no independent test or robot motion approval."}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAINING_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()