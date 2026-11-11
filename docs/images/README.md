# Project gallery provenance

These are selected copies of existing local project artifacts, published for documentation. Original images, annotations, and training outputs were not modified. No synthetic success screenshots or CVAT UI screenshots were created.

| Published image | Original local artifact | Interpretation |
|---|---|---|
| [wrist-detections.png](wrist-detections.png) | `ros2_ws/log/wrist_yolo_domain44.png` | Actual seven-class detector output from frozen preview |
| [bottle-in-tray.png](bottle-in-tray.png) | `ros2_ws/log/wrist_yolo_domain46.png` | Moving-scene camera output showing bottle in tray; not standalone proof of task completion |
| [wrist-rgb.png](wrist-rgb.png) | `ros2_ws/log/wrist_camera_rgb.png` | Raw ideal virtual-camera RGB |
| [cvat-label-preview.jpg](cvat-label-preview.jpg) | `data/datasets/yolov9_tabletop_v1/annotation_preview.jpg` | Preview generated from exported manual CVAT labels; not CVAT UI |
| [dataset-views.jpg](dataset-views.jpg) | `tableimage/zoom_views/metadata/contact_sheet_01.jpg` | Virtual detail-capture contact sheet, not training input |
| [training-curves.png](training-curves.png) | `outputs/training/tabletop_yolov9s_v1/results.png` | Recorded 100-epoch run; zero validation loss is an upstream disabled-calculation artifact |
| [confusion-matrix.png](confusion-matrix.png) | `outputs/training/tabletop_yolov9s_v1/confusion_matrix.png` | Single-scene seven-class validation confusion matrix |
| [validation-predictions.jpg](validation-predictions.jpg) | `outputs/training/tabletop_yolov9s_v1/val_batch0_pred.jpg` | Validation predictions, including errors; not manual labels |

Original artifact paths above are local provenance references and are not public download links. The gallery is not the full dataset, an independent evaluation set, or a collection of real-world photographs. Screenshots of the desktop and private CVAT account details are not included.

Model and result context: [training guide](../yolov9_training.md), [camera and detection guide](../trained_wrist_detection.md), [end-to-end workflow](../dataset_to_robot.md).