"""Same-scene wrist RGB diagnostics; no motion control or 3-D feedback.

Integration contract (all calls on the Kit/main thread): construct with the
current stage STOPPED, before controller construction; bind the existing domain
46 ROS node after controller initialization and its 120 updates; call capture()
after the outer app.update() in YOLOBottlePlaceNode.tick(); close before closing
the stage/app or destroying that node. This module does not install these hooks.

The saved hand pose determines the mount ONCE, exactly as in wrist_camera_ros.
Thereafter the child camera inherits the rendered parent transform, including
Fabric/physics motion. No per-frame USD pose sampling or world-pose writes are
used. The external observer viewport is untouched.

Verified against installed Isaac Sim 6.2 sources (under site-packages/isaacsim):
* extsDeprecated/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py:
  initialize, attach_annotator, get_current_frame, _data_acquisition_callback,
  destroy. rendering_frame is a ReferenceTime dict, NOT an integer counter.
* extsDeprecated/isaacsim.core.prims/isaacsim/core/prims/impl/xform_prim.py:
  _set_xform_properties normalizes transforms during Camera construction.
* extscache/omni.replicator.core-1.13.38+110.4.0.lx64.r.cp312/omni/replicator/
  core/scripts/{create.py,utils/viewport_manager.py}: render_product creates a
  camera-bound Hydra texture, without creating/changing a viewport.

Only completed Camera callback snapshots are published. Their render time is
simulation time; ROS headers are wall time. No depth, CameraInfo, TF or claim of
render-associated 3-D robot pose is supplied. A changing frame token does not
prove useful visibility or correct robot tracking. Moving-render behavior still
needs validation in the caller's actual scene. Recreate after a stage/reset;
never combine these images with frozen-preview domain 44 depth/TF.
"""

from collections.abc import Mapping
from fractions import Fraction
import json
import math
from numbers import Integral, Real
import time
import warnings


WIDTH, HEIGHT = 640, 480
OPTICAL = "wrist_camera_optical_frame"
DOMAIN = 46


class LiveMotionCamera:
    """Passive, at-most-5-Hz RGB publisher using an existing rclpy node."""

    def __init__(self, stage):
        # Isaac imports must happen after the caller has created SimulationApp.
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Usd, UsdGeom

        if stage is None or omni.usd.get_context().get_stage() != stage:
            raise ValueError("LiveMotionCamera requires the current Isaac stage")
        if not omni.timeline.get_timeline_interface().is_stopped():
            raise RuntimeError("Author the wrist mount while STOPPED, before the controller")
        hands = [prim for prim in stage.Traverse() if prim.GetName() == "panda_hand"]
        if len(hands) != 1:
            raise RuntimeError(f"Expected one panda_hand, found {len(hands)}")

        self._stage = stage
        self.camera_path = str(hands[0].GetPath()) + "/LiveMotionCamera"
        if stage.GetPrimAtPath(self.camera_path):
            raise RuntimeError(f"Refusing to overwrite {self.camera_path}")
        self._camera = None
        self._product = None
        self._node = None
        self._color_pub = None
        self._source_pub = None
        self._bound = False
        self._closed = False
        self._last_publish = -math.inf
        self._last_reference = None
        self._last_rendering_time = -math.inf
        self.frames_published = 0

        # Match the frozen preview's saved-hand geometry exactly, including the
        # USD row-vector matrix convention. Do not aim again after physics starts.
        hand_world = UsdGeom.XformCache().GetLocalToWorldTransform(hands[0])
        eye = hand_world.Transform(Gf.Vec3d(0.08, 0.0, 0.035))
        target = Gf.Vec3d(0.65, 0.04, 0.025)
        camera_world = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        local = camera_world * hand_world.GetInverse()
        try:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                camera = UsdGeom.Camera.Define(stage, self.camera_path)
                camera.AddTransformOp().Set(local)
                camera.CreateFocalLengthAttr(18.0)
                camera.CreateHorizontalApertureAttr(36.0)
                camera.CreateVerticalApertureAttr(36.0 * HEIGHT / WIDTH)
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 5.0))
                self._product = rep.create.render_product(
                    self.camera_path, (WIDTH, HEIGHT), force_new=True,
                    write_to_session_layer=True,
                )
                if self._product.hydra_texture is None:
                    raise RuntimeError("A live Hydra render product is required")
                # Only this owned sensor texture: no global renderer settings,
                # capture-on-play toggles, orchestrator steps or timeline changes.
                self._product.hydra_texture.set_updates_enabled(True)
                self._camera = Camera(
                    prim_path=self.camera_path, name="live_motion_camera",
                    resolution=(WIDTH, HEIGHT),
                    render_product_path=self._product.path, annotator_device="cpu",
                )
                # BaseSensor/SingleXFormPrim normalizes xform ops in __init__.
                # Restore the exact matrix now, NOT later against a moving hand.
                camera.ClearXformOpOrder()
                camera.AddTransformOp().Set(local)
                camera.SetResetXformStack(False)
        except Exception:
            self.close()
            raise

    def bind(self, node) -> None:
        """Attach RGB after controller warm-up; borrow, never own, its ROS node.

        No updates/waits are performed here. Subsequent caller-owned app updates
        render the texture; Camera's NEW_FRAME callback evaluates its SDG graph
        and stores RGB and ReferenceTime in the same current-frame dictionary.
        """
        import omni.timeline
        import omni.usd
        from pxr import Usd
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        if self._closed or self._bound:
            raise RuntimeError("Camera is closed or already bound")
        if omni.usd.get_context().get_stage() != self._stage:
            raise RuntimeError("Camera stage is no longer current")
        if node.context.get_domain_id() != DOMAIN:
            raise RuntimeError("Live motion RGB requires the existing ROS context in domain 46")
        if not omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError("Bind after controller initialization/playback and its 120 updates")
        self._node = node
        try:
            with Usd.EditContext(self._stage, self._stage.GetSessionLayer()):
                self._camera.initialize(attach_rgb_annotator=False)
                self._camera.attach_annotator("rgb")
            # Reliable depth-one QoS also serves best-effort image subscribers.
            self._color_pub = node.create_publisher(Image, "/wrist_camera/color/image_raw", 1)
            self._source_pub = node.create_publisher(String, "/wrist_camera/source", 1)
            self._bound = True
        except Exception:
            self.close()
            raise

    def capture(self) -> bool:
        """Publish a fresh RGB snapshot, or return False during warm-up/throttle.

        Call only AFTER the caller's app.update(), on the same thread. This does
        not advance rendering/physics or spin ROS. Duplicate, reversed, missing
        or malformed render metadata fails closed; no wall-time freshness proxy.
        """
        import numpy as np
        import omni.usd
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        if self._closed:
            return False
        if not self._bound:
            raise RuntimeError("Call bind(node) before capture()")
        if omni.usd.get_context().get_stage() != self._stage:
            return False
        if time.monotonic() - self._last_publish < 0.2:
            return False

        # clone=True copies callback data, not a new annotation from a different
        # render. In this SDK rgb is CPU uint8 RGBA (annotators.py registry).
        frame = self._camera.get_current_frame(clone=True)
        reference = frame.get("rendering_frame")
        rendering_time = frame.get("rendering_time")
        if not isinstance(reference, Mapping):
            return False  # initialize() leaves the placeholder integer 0.
        numerator = reference.get("referenceTimeNumerator")
        denominator = reference.get("referenceTimeDenominator")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in (numerator, denominator)):
            return False
        if numerator < 0 or denominator <= 0:
            return False
        token = Fraction(int(numerator), int(denominator))
        if (isinstance(rendering_time, bool) or not isinstance(rendering_time, Real)
                or not math.isfinite(rendering_time) or rendering_time < 0):
            return False
        if ((self._last_reference is not None and token <= self._last_reference)
                or rendering_time < self._last_rendering_time):
            return False
        rgba = frame.get("rgb")
        if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8
                or rgba.shape not in ((HEIGHT, WIDTH, 4), (HEIGHT, WIDTH, 3))):
            return False
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        seconds, nanoseconds = divmod(time.time_ns(), 1_000_000_000)
        image = Image()
        image.header.stamp.sec, image.header.stamp.nanosec = seconds, nanoseconds
        image.header.frame_id = OPTICAL
        image.height, image.width = HEIGHT, WIDTH
        image.encoding, image.is_bigendian, image.step = "rgb8", 0, WIDTH * 3
        image.data = rgb.tobytes()
        source = String()
        source.data = json.dumps({
            "source": "motion_validation", "domain": DOMAIN,
            "mode": "RGB-only moving preview", "motion_feedback_3d": False,
            "depth_published": False, "tf_published": False,
            "camera_path": self.camera_path,
            "render_product_path": self._product.path,
            "rendering_frame": {
                "referenceTimeNumerator": int(numerator),
                "referenceTimeDenominator": int(denominator),
            },
            "rendering_time": float(rendering_time),
            "rendering_time_clock": "simulation", "header_clock": "wall",
            "header": {"stamp": {"sec": seconds, "nanosec": nanoseconds}, "frame_id": OPTICAL},
        }, allow_nan=False)
        # Consume before publishing: even a ROS publish exception cannot cause
        # this same render to be stamped as a new image on a subsequent call.
        self._last_reference = token
        self._last_rendering_time = float(rendering_time)
        self._last_publish = time.monotonic()
        self._source_pub.publish(source)
        self._color_pub.publish(image)
        self.frames_published += 1
        return True

    def close(self) -> None:
        """Idempotent best-effort cleanup; never stop the node/app or physics.

        Close while this stage is current. If it has already been replaced,
        avoid deleting a same-named render product in the replacement stage.
        """
        if self._closed:
            return
        self._closed = True
        self._bound = False
        import omni.usd
        from pxr import Usd

        failures = []

        def release(label, action):
            try:
                action()
            except Exception as exc:
                failures.append(f"{label}: {exc}")

        if self._camera is not None:
            release("Camera.destroy", self._camera.destroy)
        same_stage = omni.usd.get_context().get_stage() == self._stage
        if self._product is not None:
            if same_stage:
                # Camera received an external render product, so destroy() only
                # detaches its annotators/subscriptions; we own this product.
                release("render product", self._product.destroy)
            elif self._product.hydra_texture is not None:
                release("disable old texture", lambda: self._product.hydra_texture.set_updates_enabled(False))
                failures.append("Stage replaced before close; old render product left to Kit teardown")
        if same_stage:
            def remove_mount():
                with Usd.EditContext(self._stage, self._stage.GetSessionLayer()):
                    self._stage.RemovePrim(self.camera_path)
            release("session mount", remove_mount)
        if self._node is not None:
            for publisher in (self._color_pub, self._source_pub):
                if publisher is not None:
                    release("publisher", lambda pub=publisher: self._node.destroy_publisher(pub))
        self._color_pub = self._source_pub = self._node = None
        self._camera = self._product = None
        for failure in failures:
            warnings.warn(f"LiveMotionCamera cleanup: {failure}", RuntimeWarning, stacklevel=2)