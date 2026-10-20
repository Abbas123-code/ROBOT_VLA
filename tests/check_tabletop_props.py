"""Isolated prop-settling test. Robot removed before Play; no ROS or controller."""
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac/scripts"))

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.xforms import get_world_pose
    from tabletop_props import LAYOUT, PROPS_PATH, check_layout

    context = omni.usd.get_context()
    assert context.open_stage(str(ROOT / "isaac/scenes/panda_tabletop_cluttered.usda"))
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    check_layout(stage)
    stage.RemovePrim("/World/ROS2")
    stage.RemovePrim("/World/panda")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.play()
    for _ in range(240):
        app.update()
    positions = {}
    for name, expected in LAYOUT.items():
        p, q = get_world_pose(f"{PROPS_PATH}/{name}", fabric=True)
        np.testing.assert_allclose(p, expected, atol=0.003)
        assert abs(float(q[0])) > np.cos(0.05 / 2), f"Prop tilted: {name}"
        positions[name] = np.asarray(p).tolist()
    report = {"passed": True, "objects": positions, "simulation_seconds": timeline.get_current_time(),
              "scope": "Static clearance and isolated prop stability only; no robot trajectory tested."}
    report_path = ROOT / "ros2_ws/log/tabletop_props_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    timeline.stop()
    print("TABLETOP_PROPS_TEST_PASSED " + json.dumps(report), flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close(skip_cleanup=False)