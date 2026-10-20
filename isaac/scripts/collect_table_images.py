"""Capture unlabelled table RGB images, without robot motion or saving USD.

Orbit mode: baseline wrist view plus 60 virtual viewpoints with identical optics.
Detail mode: near/medium/far object views and wider whole-table views.
Virtual viewpoints are NOT claimed to be reachable Panda wrist configurations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]


def detail_views(stage):
    """Use static scene bounds for framing only, never as training labels."""
    import numpy as np
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    objects = {"bottle": "/World/TabletopProps/WaterBottle", "can": "/World/TabletopProps/OrangeCan",
               "box": "/World/PlacementBox", "red_cube": "/World/Cube",
               "green_cube": "/World/TabletopProps/GreenCube", "black_cube": "/World/TabletopProps/BlackCube",
               "yellow_block": "/World/TabletopProps/YellowBlock"}
    views = []

    def add(target, distance, azimuth, elevation, **metadata):
        a, e = math.radians(azimuth), math.radians(elevation)
        eye = target + distance * np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
        views.append({"kind": "virtual_distance_variation", "azimuth_deg": azimuth,
                      "elevation_deg": elevation, "distance_m": float(distance),
                      "eye": eye.tolist(), "target": target.tolist(), **metadata})

    for name, path in objects.items():
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if bounds.IsEmpty() or not np.isfinite([lo, hi]).all():
            raise RuntimeError(f"Invalid object framing bounds: {path}")
        target = (lo + hi) / 2
        # Bounding sphere fits inside the narrower vertical FOV with margin.
        near = max(.16, 2.25 * float(np.linalg.norm((hi - lo) / 2)))
        for angle, elevation in ((20, 35), (140, 55), (260, 40)):
            for scale, multiplier in (("near", 1.), ("medium", 1.65), ("far", 2.6)):
                add(target, near * multiplier, angle, elevation, focus_object=name, scale=scale,
                    filename=f"zoom_{name}_a{angle:03d}_{scale}.png")
    for angle in range(0, 360, 60):
        for scale, distance in (("wide", 1.05), ("wider", 1.35)):
            add(np.array([.65, .04, .07]), distance, angle, 55, focus_object="table", scale=scale,
                filename=f"zoom_table_a{angle:03d}_{scale}.png")
    return views


def collect(app, output, mode="orbit"):
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image, ImageDraw
    from pxr import Gf, UsdGeom
    from wrist_camera_ros import SCENE, WIDTH, HEIGHT, FOCAL, H_APERTURE

    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError("Could not open saved clutter scene")
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    source_hashes = {Path(layer.realPath): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                     for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    stage.SetEditTarget(stage.GetSessionLayer())
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    graph = stage.GetPrimAtPath("/World/ROS2")
    if graph:
        graph.SetActive(False)
    hands = [p for p in stage.Traverse() if p.GetName() == "panda_hand"]
    if len(hands) != 1:
        raise RuntimeError("Expected a single Panda wrist")
    hand_world = UsdGeom.XformCache().GetLocalToWorldTransform(hands[0])
    tracked = ["/World/panda", "/World/Cube", "/World/PlacementBox"]
    tracked += [f"/World/TabletopProps/{name}" for name in
                ("GreenCube", "BlackCube", "WaterBottle", "OrangeCan", "YellowBlock")]
    cache = UsdGeom.XformCache()
    original_poses = {path: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) for path in tracked}
    camera = UsdGeom.Camera.Define(stage, "/World/DatasetCamera")
    transform = camera.AddTransformOp()
    camera.CreateFocalLengthAttr(FOCAL)
    camera.CreateHorizontalApertureAttr(H_APERTURE)
    camera.CreateVerticalApertureAttr(H_APERTURE * HEIGHT / WIDTH)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 5.))
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(str(camera.GetPath()), (WIDTH, HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    eye = hand_world.Transform(Gf.Vec3d(.08, 0., .035))
    views = [{"kind": "baseline_wrist_mount", "eye": list(eye), "target": [.65, .04, .025]}]
    target = np.array([.65, .04, .07])
    # Full 360-degree coverage at three elevations. A modest distance change
    # also varies object scale; all scene objects and lighting remain unchanged.
    for elevation, radius in ((30, .85), (50, .75), (70, .65)):
        for azimuth in range(0, 360, 18):
            a, e = math.radians(azimuth), math.radians(elevation)
            eye = target + radius * np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
            views.append({"kind": "virtual_multiview", "azimuth_deg": azimuth,
                          "elevation_deg": elevation, "distance_m": radius,
                          "eye": eye.tolist(), "target": target.tolist()})
    if mode == "detail":
        views = detail_views(stage)
    output.mkdir(parents=True, exist_ok=True)
    meta = output / "metadata"
    meta.mkdir(exist_ok=True)
    report = {"completed": False, "scene": str(SCENE.relative_to(ROOT)),
              "resolution": [WIDTH, HEIGHT], "focal_length_mm": FOCAL,
              "horizontal_aperture_mm": H_APERTURE,
              "K": [WIDTH*FOCAL/H_APERTURE, 0, WIDTH/2, 0, WIDTH*FOCAL/H_APERTURE, HEIGHT/2, 0, 0, 1],
              "robot_motion": False, "physics_time": 0,
              "mode": mode,
              "scope": "Virtual camera views with wrist optics; not verified reachable wrist poses. Orbit includes baseline wrist view.",
              "labels": "None. Original RGB images for manual annotation; no YOLO predictions burned in.",
              "frames": []}
    try:
        hashes = set()
        for index, view in enumerate(views):
            world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*view["eye"]), Gf.Vec3d(*view["target"]),
                                          Gf.Vec3d(0, 0, 1)).GetInverse()
            transform.Set(world)
            # Two captures per new view flush stale buffers and settle rendering.
            for _ in range(2):
                rep.orchestrator.step(delta_time=0., pause_timeline=True, rt_subframes=4)
            rgba = np.asarray(rgb.get_data())
            if rgba.shape[:2] != (HEIGHT, WIDTH) or rgba.ndim != 3:
                raise RuntimeError(f"Missing render buffer at view {index}: {rgba.shape}")
            color = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
            if color.std() < 5:
                raise RuntimeError(f"Blank image at view {index}")
            digest = hashlib.sha256(color.tobytes()).hexdigest()
            if digest in hashes:
                raise RuntimeError(f"Duplicate pixels at view {index}")
            hashes.add(digest)
            cache = UsdGeom.XformCache()
            if abs(timeline.get_current_time()) > 1e-8 or timeline.is_playing():
                raise RuntimeError("Physics unexpectedly advanced")
            if not Gf.IsClose(cache.GetLocalToWorldTransform(hands[0]), hand_world, 1e-8):
                raise RuntimeError("Robot wrist moved")
            for path, pose in original_poses.items():
                if not Gf.IsClose(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), pose, 1e-8):
                    raise RuntimeError(f"Scene object moved: {path}")
            name = view.get("filename", f"table_{index:03d}.png")
            path = output / name
            with path.open("xb") as stream:
                Image.fromarray(color).save(stream, format="PNG")
            record = {"file": name, **view, "pixel_sha256": digest,
                      "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "camera_world_usd_row_matrix": np.asarray(world).tolist()}
            report["frames"].append(record)
            print(f"TABLE_IMAGE {index+1}/{len(views)} {name} {view['kind']}", flush=True)
        report["source_layers_unchanged"] = all(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
        if not report["source_layers_unchanged"]:
            raise RuntimeError("Source USD layers changed on disk")
        report["image_count"] = len(report["frames"])
        # Contact sheets belong in metadata, not among the annotation images.
        for start in range(0, len(views), 20):
            subset = report["frames"][start:start+20]
            sheet = Image.new("RGB", (4*240, math.ceil(len(subset)/4)*200), "#20252b")
            draw = ImageDraw.Draw(sheet)
            for n, frame in enumerate(subset):
                with Image.open(output / frame["file"]) as source:
                    thumb = source.resize((240, 180))
                x, y = (n % 4)*240, (n // 4)*200
                sheet.paste(thumb, (x, y))
                draw.text((x+5, y+182), frame["file"], fill="white")
            sheet.save(meta / f"contact_sheet_{start//20+1:02d}.jpg", quality=92)
        report["completed"] = True
        print(f"TABLE_IMAGES_COMPLETE {len(views)} unique images in {output}", flush=True)
    finally:
        (meta / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        timeline.pause()
        rgb.detach(product)
        product.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("orbit", "detail"), default="orbit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "tableimage" if args.mode == "orbit" else ROOT / "tableimage/zoom_views"
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Output directory must be empty; refusing to overwrite existing images or labels")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        collect(app, args.output, args.mode)
    except BaseException:
        traceback.print_exc()
        print("TABLE_IMAGES_FAILED", flush=True)
        raise
    finally:
        app.close(skip_cleanup=False)


if __name__ == "__main__":
    main()