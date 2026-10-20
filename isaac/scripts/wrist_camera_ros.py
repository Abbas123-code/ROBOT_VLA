"""Render a wrist RGB-D camera to ROS 2; frozen scene, no motion controller.

Source ROS Jazzy before running. --validate captures five frames and exits.
Edits exist only in the USD session layer; existing scenes are never saved.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "isaac/scenes/panda_tabletop_cluttered.usda"
REPORTS = ROOT / "ros2_ws/log"
WIDTH, HEIGHT = 640, 480
FOCAL, H_APERTURE = 18.0, 36.0
OPTICAL = "wrist_camera_optical_frame"
PREFIX = "/wrist_camera"


def run(app, validate):
    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Usd, UsdGeom

    enable_extension("isaacsim.ros2.bridge")
    enable_extension("omni.kit.window.stage")
    app.update()
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import TransformBroadcaster

    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError(f"Could not open {SCENE}")
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    layers = {Path(layer.realPath): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    stage.SetEditTarget(stage.GetSessionLayer())
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    stage.GetPrimAtPath("/World/ROS2").SetActive(False)
    hands = [prim for prim in stage.Traverse() if prim.GetName() == "panda_hand"]
    if len(hands) != 1:
        raise RuntimeError(f"Expected one panda_hand, found {len(hands)}")
    hand = hands[0]
    hand_world = UsdGeom.XformCache().GetLocalToWorldTransform(hand)
    # A side-mounted eye avoids placing the lens inside the fingers. Aim once,
    # then author a fixed LOCAL mount, so the camera inherits the wrist frame.
    eye = hand_world.Transform(Gf.Vec3d(0.08, 0.0, 0.035))
    target = Gf.Vec3d(0.65, 0.04, 0.025)
    camera_world = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    local = camera_world * hand_world.GetInverse()
    camera_path = str(hand.GetPath()) + "/WristCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddTransformOp().Set(local)
    camera.CreateFocalLengthAttr(FOCAL)
    camera.CreateHorizontalApertureAttr(H_APERTURE)
    camera.CreateVerticalApertureAttr(H_APERTURE * HEIGHT / WIDTH)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 5.0))
    # A sensor-only mount: no housing collider or extra mass on the robot.
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(camera_path, (WIDTH, HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb.attach(product)
    depth.attach(product)
    carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — Wrist RGB-D (frozen scene, domain 44)")
    if not validate:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(camera_path)

    rclpy.init(args=[])
    node = rclpy.create_node("wrist_camera_publisher")
    # Reliable depth-1 publishers also accept best-effort RViz sensor subscribers.
    color_pub = node.create_publisher(Image, PREFIX + "/color/image_raw", 1)
    depth_pub = node.create_publisher(Image, PREFIX + "/depth/image_raw", 1)
    info_pub = node.create_publisher(CameraInfo, PREFIX + "/color/camera_info", 1)
    depth_info_pub = node.create_publisher(CameraInfo, PREFIX + "/depth/camera_info", 1)
    tf = TransformBroadcaster(node)
    optical_flip = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180))
    fx = WIDTH * FOCAL / H_APERTURE
    info = CameraInfo(width=WIDTH, height=HEIGHT, distortion_model="plumb_bob")
    info.d = [0.0] * 5
    info.k = [fx, 0., WIDTH / 2, 0., fx, HEIGHT / 2, 0., 0., 1.]
    info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    info.p = [fx, 0., WIDTH / 2, 0., 0., fx, HEIGHT / 2, 0., 0., 0., 1., 0.]

    def transform(parent, child, matrix, stamp):
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id, msg.child_frame_id = parent, child
        p = matrix.ExtractTranslation()
        q = matrix.ExtractRotationQuat()
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, p)
        msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z = map(float, q.GetImaginary())
        msg.transform.rotation.w = float(q.GetReal())
        return msg

    def image(array, encoding, stamp):
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, OPTICAL
        msg.height, msg.width = HEIGHT, WIDTH
        msg.encoding, msg.is_bigendian = encoding, 0
        msg.step = WIDTH * (3 if encoding == "rgb8" else 4)
        msg.data = array.tobytes()
        return msg

    count = 0
    attempts = 0
    last_publish = 0.0
    try:
        print(f"WRIST_CAMERA_CAPTURE_START: {camera_path}, eye={eye}", flush=True)
        while app.is_running():
            if timeline.is_playing():
                timeline.pause()
                raise RuntimeError("Preview must stay paused; no physics playback is authorized here")
            if not validate and time.monotonic() - last_publish < 0.2:
                app.update()
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            # Installed Replicator explicitly supports captures at a fixed time.
            rep.orchestrator.step(delta_time=0.0, pause_timeline=True, rt_subframes=2)
            attempts += 1
            rgba = rgb.get_data()
            z = depth.get_data()
            if validate:
                print(f"WRIST_CAMERA_FRAME: attempt={attempts}, RGB={rgba.shape}, depth={z.shape}", flush=True)
            if rgba.shape[:2] != (HEIGHT, WIDTH) or z.shape != (HEIGHT, WIDTH):
                if attempts > 30:
                    raise RuntimeError(f"Missing camera buffers: RGB {rgba.shape}, depth {z.shape}")
                continue
            color = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
            z = np.asarray(z, dtype="<f4").copy()
            valid = np.isfinite(z) & (z >= 0.02) & (z <= 5.0)
            if np.count_nonzero(valid) < WIDTH * HEIGHT * 0.1 or color.std() < 5:
                raise RuntimeError("Camera image is blank or depth mostly invalid")
            z[~valid] = np.nan
            stamp = node.get_clock().now().to_msg()
            # Viewport navigation can edit the camera, even while physics is
            # stopped. Publish its actual current pose and intrinsics, not the
            # initial mount values. RGB-D consumers rely on this calibration.
            cache = UsdGeom.XformCache()
            current_hand = cache.GetLocalToWorldTransform(hand)
            current_camera = cache.GetLocalToWorldTransform(camera.GetPrim())
            optical_local = optical_flip * current_camera * current_hand.GetInverse()
            focal = float(camera.GetFocalLengthAttr().Get())
            horizontal = float(camera.GetHorizontalApertureAttr().Get())
            vertical = float(camera.GetVerticalApertureAttr().Get())
            fx, fy = WIDTH * focal / horizontal, HEIGHT * focal / vertical
            cx = WIDTH / 2 - WIDTH * float(camera.GetHorizontalApertureOffsetAttr().Get()) / horizontal
            cy = HEIGHT / 2 + HEIGHT * float(camera.GetVerticalApertureOffsetAttr().Get()) / vertical
            info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
            info.p = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
            info.header.stamp, info.header.frame_id = stamp, OPTICAL
            color_pub.publish(image(color, "rgb8", stamp))
            depth_pub.publish(image(z, "32FC1", stamp))
            info_pub.publish(info)
            depth_info_pub.publish(info)
            if not Gf.IsClose(current_hand, hand_world, 1e-8) or abs(timeline.get_current_time()) > 1e-8:
                raise RuntimeError("Frozen wrist pose or simulation time changed")
            tf.sendTransform([transform("world", "panda_hand", current_hand, stamp),
                              transform("panda_hand", OPTICAL, optical_local, stamp)])
            rclpy.spin_once(node, timeout_sec=0.0)
            last_publish = time.monotonic()
            count += 1
            if count == 5:
                from PIL import Image as PillowImage
                REPORTS.mkdir(parents=True, exist_ok=True)
                PillowImage.fromarray(color).save(REPORTS / "wrist_camera_rgb.png")
                unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in layers.items())
                assert unchanged, "Source layer changed on disk"
                report = {"completed": True, "domain": 44, "camera_path": camera_path,
                          "frames": count, "resolution": [WIDTH, HEIGHT],
                          "depth_encoding": "32FC1", "depth_units": "metres, optical-axis Z",
                          "valid_depth_fraction": float(valid.mean()),
                          "depth_range_m": [float(z[valid].min()), float(z[valid].max())],
                          "camera_world_position": list(eye), "fx": fx,
                          "source_layers_unchanged": unchanged, "simulation_time": timeline.get_current_time(),
                          "robot_motion": False, "mode": "validation" if validate else "live_preview"}
                name = "wrist_camera_validation.json" if validate else "wrist_camera_preview.json"
                (REPORTS / name).write_text(json.dumps(report, indent=2) + "\n")
                print("WRIST_CAMERA_VALIDATED" if validate else "WRIST_CAMERA_READY: RGB + depth + CameraInfo + TF, domain 44, no motion", flush=True)
                if validate:
                    break
        if count < 5:
            raise RuntimeError(f"Camera stopped before validation: {count} valid frames, {attempts} attempts")
    finally:
        timeline.pause()
        rgb.detach(product)
        depth.detach(product)
        product.destroy()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = "44"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "ros2_ws/.wrist_camera.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.validate})
        try:
            run(app, args.validate)
        except BaseException:
            traceback.print_exc()
            print("WRIST_CAMERA_FAILED", flush=True)
            raise
        finally:
            app.close(skip_cleanup=False)


if __name__ == "__main__":
    main()