from pathlib import Path
import runpy
import traceback

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.timeline
    import omni.usd

    context = omni.usd.get_context()
    if not context.open_stage(str(ROOT / "isaac/scenes/panda_tabletop.usda")):
        raise RuntimeError("Could not open the saved test scene")
    for _ in range(10):
        app.update()
    # Remove ROS only from this unsaved test stage to avoid traffic to the user's simulation.
    context.get_stage().RemovePrim("/World/ROS2")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
    namespace = runpy.run_path(str(ROOT / "isaac/scripts/inspect_live_positions.py"))
    report = namespace["inspect_positions"]()
    np.testing.assert_allclose(report["base_xyz"], [0, 0, 0], atol=1e-4)
    assert 0.015 < report["cube_center_xyz"][2] < 0.035, report
    assert 0.4 < report["grasp_target_xyz"][2] < 0.6, report
    assert np.isfinite(report["distance_metres"])
    print("PASS: runtime pose inspector reads base, gripper, and settled cube in an isolated scene", flush=True)
    timeline.stop()
    for _ in range(10):
        app.update()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close(skip_cleanup=False)