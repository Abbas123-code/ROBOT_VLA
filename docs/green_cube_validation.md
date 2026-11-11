# Green cube beside bottle: blocked validation

User authorized a reconstructed scene with the bottle initialized at its recorded
placed position, not a continuation of the completed live bottle process.
Original bottle window and saved USD scenes remain untouched. Domain 47 isolates
the new visible validation. Targeting uses simulator-known coordinates, NOT YOLO.

Entry point: `isaac/scripts/run_green_cube.py`; controller:
`isaac/scripts/green_cube_controller.py`; pure geometry/IK:
`src/control/green_cube_plan.py`. The launcher requires explicit validation or
execution; execution checks a completed matching fingerprint. No passing green
validation exists. The visible task is in `.vscode/tasks.json`.

## Observed result

The latest individually authorized visible trial gripped the green cube with
loaded bilateral finger contact, lifted it and transferred it over the tray.
It aborted during lowering, before release, at simulation time 117.0667 s.
The palm/bottle-body pair reported separation 0.01083950698 m and zero impulse.
This is proximity, not evidence of physical collision. Palm/obstacle proximity
remains forbidden: do not weaken that guard to force placement.

The new scene is intentionally paused with the cube held. No automatic release,
retry, teleport or attachment occurred. Report:
`ros2_ws/log/green_cube_validation.json`, `completed: false`.

Earlier trials exposed narrow proximity-policy mistakes during finger closure,
table liftoff and cube/bottle placement. Regression-tested exceptions require
at least 2 mm separation and EXACT zero impulse. They never count as grip or
floor support. Other contacts, slip, tracking, tilt and neighbour checks remain.
Latest controller tests: 21 passed; geometry/IK tests: 9 passed. Nominal route:
12 segments, peak step 0.010617 rad, exact-HOME correction 0.035723 rad.

## Clearance investigation: no approved replacement route

Read-only numeric inspection used the actual URDF and USD hand/finger collision
meshes, matching transformed vertices, bottle cylinder/sphere geometry and tray
walls. Green root is the BOTTOM at baseline (0.70, 0.14, 0), not its centre.
Bottle reconstruction root: (0.4466993213, -0.2437739223, 0.0079996884) m.

The current target (0.512, -0.284, 0.008) fits the cube footprint, but not the full
gripper opening envelope: opening to 0.040 m would intersect a side wall.
The opposite corner also has a finger/wall intersection. This demonstrates that
the existing object-only footprint preflight is NOT a gripper collision proof.

With the measured carry offset and at least 5 mm nominal cube/rim margin, a
read-only target/yaw search at 0/90/180/270 degrees found no candidate meeting
12 mm hand/bottle clearance while allowing full opening. Conservative best-gap
upper bounds were 3.535/8.614/1.726/7.654 mm respectively. These are constrained
numeric results, not proof that every possible grasp/orientation is impossible.

A rejected 90-degree candidate at (0.474, -0.312, 0.008) passed full-route IK,
but conservative hand/bottle clearance was only 7.549 mm and opening/wall
clearance 5.826 mm, before dynamic tracking errors. No candidate was installed
or executed. Increasing force cannot solve this geometric constraint.

Next work requires a different grasp/approach with full gripper swept-volume
validation, or explicit permission to change the bottle position or tray.
Do not resume the paused trial or execute the existing route as approved.