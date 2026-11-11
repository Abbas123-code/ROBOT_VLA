"""Validate a CVAT COCO export and create an independent YOLO detection copy."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yolo_box(bbox, width, height):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError(f"Invalid box: {bbox}")
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"Box outside image or non-positive: {bbox}")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def view_group(name, frames):
    match = re.search(r"_a(\d{3})_", name)
    angle = int(match[1]) if match else frames.get(name, {}).get("azimuth_deg")
    # Keep near/medium/far, wide/wider and orbit elevations in the same sector.
    return f"azimuth_sector_{int(angle) // 60}" if angle is not None else f"view_{name}"


def choose_split(records, class_ids, fraction=0.2):
    groups = sorted({r["group"] for r in records})
    if len(groups) > 18:
        raise ValueError("Too many groups for exhaustive split; define coarser view groups")
    best = None
    for n in range(1, len(groups)):
        for selected in itertools.combinations(groups, n):
            val = [r for r in records if r["group"] in selected]
            train = [r for r in records if r["group"] not in selected]
            if any({a["category_id"] for r in split for a in r["annotations"]} != class_ids
                   for split in (train, val)):
                continue
            score = (abs(len(val) / len(records) - fraction), selected)
            if best is None or score < best[0]:
                best = (score, train, val)
    if best is None:
        raise ValueError("Cannot make a disjoint grouped split containing every class")
    return best[1], best[2]


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    annotation_file = source / "labels/instances_default.json"
    data = json.loads(annotation_file.read_text())
    categories = sorted(data["categories"], key=lambda c: c["id"])
    ids = [c["id"] for c in categories]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Category IDs must be unique and nonempty")
    class_map = {category_id: i for i, category_id in enumerate(ids)}
    names = [c["name"] for c in categories]
    images = {im["id"]: im for im in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate COCO image IDs")
    annotations = defaultdict(list)
    annotation_ids = set()
    for ann in data["annotations"]:
        if ann["id"] in annotation_ids:
            raise ValueError("Duplicate annotation IDs")
        annotation_ids.add(ann["id"])
        if ann["image_id"] not in images or ann["category_id"] not in class_map:
            raise ValueError("Annotation refers to unknown image or category")
        if ann.get("iscrowd", 0) or ann.get("attributes", {}).get("rotation", 0):
            raise ValueError("Crowd/rotated annotations need explicit review")
        im = images[ann["image_id"]]
        yolo_box(ann["bbox"], im["width"], im["height"])
        annotations[ann["image_id"]].append(ann)
    review = json.loads((ROOT / "tableimage/zoom_views/metadata/quality_review.json").read_text())
    exclusions = {r["file"]: r["reason"] for r in review["exclude_from_training"]}
    frames = {r["file"]: r for r in json.loads(
        (ROOT / "tableimage/metadata/manifest.json").read_text())["frames"]}
    records, excluded, pixels, filenames = [], [], set(), set()
    for im in data["images"]:
        name = im["file_name"]
        if Path(name).name != name or Path(name).stem in filenames:
            raise ValueError(f"Unsafe or duplicate image stem: {name}")
        filenames.add(Path(name).stem)
        path = source / "images" / name
        with Image.open(path) as image:
            image.load()
            if image.size != (im["width"], im["height"]):
                raise ValueError(f"Image dimensions differ from COCO: {name}")
            pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if pixel_hash in pixels:
            raise ValueError(f"Duplicate image pixels require review: {name}")
        pixels.add(pixel_hash)
        if name in exclusions:
            excluded.append({"file": name, "reason": exclusions[name]})
            continue
        records.append({**im, "annotations": annotations[im["id"]],
                        "group": view_group(name, frames), "sha256": sha256(path)})
    train, val = choose_split(records, set(ids))
    report = {"source_annotations_sha256": sha256(annotation_file),
              "class_mapping": [{"coco_id": c["id"], "yolo_id": i, "name": c["name"]}
                                for i, c in enumerate(categories)],
              "excluded": excluded, "splits": {},
              "warning": "Single synthetic scene; validation is not a real-world or independent test score."}
    output.mkdir(parents=True)
    for split_name, split in (("train", train), ("val", val)):
        image_dir, label_dir = output / "images" / split_name, output / "labels" / split_name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        counts = Counter()
        for record in split:
            name = record["file_name"]
            shutil.copy2(source / "images" / name, image_dir / name)
            lines = []
            for ann in record["annotations"]:
                class_id = class_map[ann["category_id"]]
                box = yolo_box(ann["bbox"], record["width"], record["height"])
                lines.append(f"{class_id} " + " ".join(f"{v:.8f}" for v in box))
                counts[names[class_id]] += 1
            (label_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        report["splits"][split_name] = {
            "images": len(split), "boxes": sum(counts.values()), "class_counts": dict(counts),
            "groups": sorted({r["group"] for r in split}),
            "files": [{"file": r["file_name"], "sha256": r["sha256"]} for r in split]}
    (output / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(output), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": names}, sort_keys=False))
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    samples = [next((r for r in records if r["file_name"] == name), records[0]) for name in
               ("table_000.png", "zoom_bottle_a020_near.png", "zoom_can_a020_near.png", "zoom_table_a120_wide.png")]
    sheet = Image.new("RGB", (1280, 960))
    palette = ["cyan", "red", "magenta", "lime", "yellow", "orange", "white"]
    for i, record in enumerate(samples):
        image = Image.open(source / "images" / record["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in record["annotations"]:
            x, y, w, h = ann["bbox"]
            cid = class_map[ann["category_id"]]
            draw.rectangle((x, y, x+w, y+h), outline=palette[cid % len(palette)], width=2)
            draw.text((x, y), names[cid], fill=palette[cid % len(palette)], stroke_width=1, stroke_fill="black")
        draw.text((4, 4), record["file_name"], fill="white", stroke_width=1, stroke_fill="black")
        sheet.paste(image.resize((640, 480)), ((i % 2)*640, (i // 2)*480))
    sheet.save(output / "annotation_preview.jpg")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))
    print(json.dumps({s: {k: v for k, v in d.items() if k != "files"}
                      for s, d in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "data/datasets/yolov9_tabletop_v1")
    args = parser.parse_args()
    prepare(args.source, args.output)