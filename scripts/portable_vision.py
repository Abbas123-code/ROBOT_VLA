"""Portable offline image tools. No ROS publishers or robot commands."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def doctor():
    import torch
    import torchvision
    from perception.yolov9_official import OfficialYOLOv9
    model = OfficialYOLOv9()
    assert torch.version.cuda is None, "Expected CPU-only torch"
    assert importlib.util.find_spec("ultralytics") is None, "Ultralytics is not permitted"
    print(json.dumps({"completed": True, "torch": torch.__version__, "torchvision": torchvision.__version__,
                      "device": "cpu", "repository": model.config["repository"],
                      "revision": model.config["revision"], "weights_sha256": model.config["weights_sha256"]}, indent=2))


def inspect(folder):
    from PIL import Image
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and "metadata" not in p.relative_to(folder).parts)
    if not files:
        raise ValueError(f"No input images found in {folder}")
    duplicates, seen, sizes = [], {}, set()
    for path in files:
        with Image.open(path) as image:
            image.load()
            sizes.add(image.size)
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if digest in seen:
            duplicates.append([str(seen[digest]), str(path)])
        seen[digest] = path
    excluded = []
    for review in folder.rglob("quality_review.json"):
        for entry in json.loads(review.read_text()).get("exclude_from_training", []):
            excluded.append(str(review.parent.parent / entry["file"]))
    print(json.dumps({"images": len(files), "resolutions": sorted(sizes), "duplicate_pixels": duplicates,
                      "flagged_for_exclusion": excluded, "labels_modified": False}, indent=2))


def detect(path, output):
    import numpy as np
    from PIL import Image, ImageDraw
    from perception.yolov9_official import OfficialYOLOv9
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; refusing to overwrite {output}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    model = OfficialYOLOv9()
    detections = model.detect(np.asarray(image))
    draw = ImageDraw.Draw(image)
    for item in detections:
        box = item["bbox_xyxy"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1]-12)), f'{item["label"]} {item["confidence"]:.2f}', fill="red")
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / "detections.png")
    report = {"input": str(path), "objects": detections, "config": model.config,
              "warning": "Model predictions, not ground-truth labels. No motion commanded."}
    (output / "detections.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Verify CPU environment and official model")
    inventory = commands.add_parser("inspect", help="Read-only image integrity inventory")
    inventory.add_argument("folder", type=Path)
    inference = commands.add_parser("detect", help="Annotate one image with model predictions")
    inference.add_argument("image", type=Path)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "inspect":
        inspect(args.folder)
    else:
        detect(args.image, args.output)


if __name__ == "__main__":
    main()