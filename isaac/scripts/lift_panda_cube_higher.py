"""Paste into Isaac Script Editor after the successful initial 2 cm lift.

Loads the tested controller from this workspace; raises the held cube another
5 cm over 10 seconds, then holds for 2 seconds. Run once, with demo motion off.
"""
import asyncio
import os
from pathlib import Path
import runpy

# Script Editor paste has no reliable __file__; set ROBOT_VLA_ROOT or launch
# Isaac from the repository root. Loading this as a file uses its own location.
ROOT = (Path(__file__).resolve().parents[2] if "__file__" in globals()
        else Path(os.environ.get("ROBOT_VLA_ROOT", Path.cwd())))
CONTROLLER = ROOT / "isaac/scripts/lift_panda_cube.py"
if not CONTROLLER.is_file():
    raise RuntimeError("Set ROBOT_VLA_ROOT to this checkout before running the legacy lift helper")

if "panda_lift_task" in globals() and not panda_lift_task.done():
    raise RuntimeError("A lift is already running")

controller = runpy.run_path(str(CONTROLLER), run_name="panda_higher_lift_controller")
panda_lift_task = asyncio.ensure_future(controller["lift_cube"](higher=True))


def report_higher_lift(task):
    try:
        task.result()
    except BaseException as error:
        print(f"HIGHER LIFT ABORTED (simulation paused if motion started): {error}", flush=True)


panda_lift_task.add_done_callback(report_higher_lift)