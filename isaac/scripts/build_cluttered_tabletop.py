"""Build a separate clutter scene, or --preview it without automatic playback."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "isaac/scenes/panda_tabletop_with_box.usda"
OUTPUT = ROOT / "isaac/scenes/panda_tabletop_cluttered.usda"


def build():
    from pxr import Usd, UsdGeom
    from tabletop_props import add_props, check_layout

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    baseline = Usd.Stage.Open(str(SOURCE))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        SOURCE, ROOT / "isaac/scenes/panda_tabletop.usda")}
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(SOURCE)]
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(36000)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetPrimAtPath("/World/ROS2").SetActive(False)
    paths = add_props(stage)
    bounds = check_layout(stage)
    if not stage.GetRootLayer().Export(str(OUTPUT)):
        raise RuntimeError("Could not export cluttered scene")
    reopened = Usd.Stage.Open(str(OUTPUT))
    check_layout(reopened)
    assert not reopened.GetPrimAtPath("/World/ROS2").IsActive()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    print(json.dumps({"scene": str(OUTPUT), "objects": paths, "bounds": bounds,
                      "original_scenes_unchanged": True, "robot_motion_validated": False}, indent=2), flush=True)
    print("CLUTTER_SCENE_CREATED", flush=True)


def preview(app):
    import carb
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.core.utils.stage import open_stage

    enable_extension("omni.kit.window.stage")
    if not open_stage(str(OUTPUT)):
        raise RuntimeError("Clutter scene not found; build it first")
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    assert not stage.GetPrimAtPath("/World/ROS2").IsActive()
    omni.timeline.get_timeline_interface().stop()
    carb.settings.get_settings().set("/app/window/title", "ROBOT_VLA — Cluttered Table Preview (no controller)")
    set_camera_view(eye=[1.6, 1.45, 1.25], target=[0.47, 0.02, 0.06])
    print("CLUTTER_PREVIEW_READY: stopped, ROS disabled, no robot controller", flush=True)
    while app.is_running():
        app.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.preview})
    try:
        if args.preview:
            preview(app)
        else:
            build()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close(skip_cleanup=False)