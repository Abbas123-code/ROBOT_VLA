# Table images for manual annotation

Captured on 2026-10-20. **61 unique, verified RGB PNG images, 640 × 480**.
Only the `table_*.png` files are annotation inputs. The `metadata` directory
contains contact-sheet previews and capture metadata, not training images.

**Additional batch:** [zoom_views](zoom_views/README.md) contains 75 new near,
medium, far and whole-table PNGs, bringing the total to **136 images**. Three
new views are flagged for exclusion because the robot blocks the camera; consult
that batch's quality review before importing. The original 61 remain unchanged.

## What was captured

- `table_000.png`: the saved scene's baseline wrist-mount camera viewpoint.
- `table_001.png`–`table_020.png`: 20 azimuths, 18° apart, elevation 30°.
- `table_021.png`–`table_040.png`: 20 azimuths, elevation 50°.
- `table_041.png`–`table_060.png`: 20 azimuths, elevation 70°.
- All images use the wrist camera's optics: focal length 18 mm, horizontal
  aperture 36 mm. Distances for the three rings: 0.85, 0.75 and 0.65 m.

The orbit views use a **virtual camera**, not an articulated robot trajectory.
Their poses have NOT been verified as reachable or collision-free wrist poses.
The robot and objects stayed still, physics time remained zero, and source USD
files were unchanged. No GPU configuration changes were made.

## Manual labelling

Import the 61 PNGs into your annotation tool. Do not import contact sheets.
There are no model predictions, generated bounding boxes or placeholder labels.

Suggested classes for all table objects (choose and retain a consistent order):
`bottle`, `can`, `box`, `red_cube`, `green_cube`, `black_cube`, `yellow_block`.
For a bottle-only detector, use just `bottle` and label every visible bottle.
Do not mix a bottle-only class map with the seven-class map.

- Draw tight boxes; include the bottle cap as part of the bottle.
- Apply one consistent annotation policy for partially occluded objects.
- Do not guess labels for completely hidden objects. Inspect `table_011.png`
  especially: the robot blocks much of that view.
- Export in **YOLO detection format**, preserving image stems and class names.
  Keep the class-ID mapping with the export.
- Review labels before training. These 61 views of one layout are a starter
  dataset, not evidence of general recognition performance. Further collection
  should vary layout, lighting and relevant real wrist viewpoints.
- Avoid randomly splitting near-identical adjacent views across train/test.
  Prefer independent capture/layout sessions for final validation/test data.

## Previews and provenance

- [Views 000–019](metadata/contact_sheet_01.jpg)
- [Views 020–039](metadata/contact_sheet_02.jpg)
- [Views 040–059](metadata/contact_sheet_03.jpg)
- [View 060](metadata/contact_sheet_04.jpg)
- [Camera poses, intrinsics and checksums](metadata/manifest.json)
- [Collection script](../isaac/scripts/collect_table_images.py)

No training has been started. The collector refuses a nonempty output directory
to protect these images and any manual labels you add.