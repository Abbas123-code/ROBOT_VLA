# Additional close-up and zoomed-out views

75 new RGB PNG images at 640 × 480. The original 61 images in the parent
directory are unchanged. All 136 images were checked for readable RGB data,
unique pixel hashes and matching capture checksums.

## Capture coverage

- Seven objects: bottle, can, box, red cube, green cube, black cube, yellow block.
- Each object: three directions × three distances = **9 images**.
- 21 near, 21 medium, 21 far, plus 12 wide/wider whole-table images.
- `near`, `medium`, `far` filenames identify the distance setting.

This is camera-distance variation, not digital enlargement or a changed lens.
The camera uses the same optics as the wrist camera. The robot did not move;
these virtual-camera locations are not verified physical wrist poses. Scene
files and GPU settings were not changed. No labels or training were generated.

## Important quality exclusions

**72 images are recommended for manual annotation. Skip these three captures:**

- [zoom_box_a140_far.png](zoom_box_a140_far.png)
- [zoom_table_a180_wide.png](zoom_table_a180_wide.png)
- [zoom_table_a180_wider.png](zoom_table_a180_wider.png)

These views intersect or are heavily blocked by robot geometry. They are kept
for traceability, not recommended as training inputs. See the machine-readable
[quality review](metadata/quality_review.json).

Label all visible instances of your selected classes, not just the object named
in the filename. Close-up views may also show other objects partly cut off at the
image edge; use a consistent annotation policy. Do not label hidden objects.
Import PNGs only, never the contact sheets. Use the same class map as the first
batch and keep near/medium/far siblings together when making train/test splits.

## Previews

- [Bottle and can close-ups](metadata/contact_sheet_01.jpg)
- [Box and cube views](metadata/contact_sheet_02.jpg)
- [Cube and block views](metadata/contact_sheet_03.jpg)
- [Zoomed-out table views](metadata/contact_sheet_04.jpg)
- [Capture metadata](metadata/manifest.json)