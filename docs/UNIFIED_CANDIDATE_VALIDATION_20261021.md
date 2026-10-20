# Unified YOLOv11 candidate — evaluation gate (21 October 2026)

The completed Google Colab export arrived as `CircuitVision_Unified_YOLO11_Candidate.zip`.
It contains `best.onnx` (10.75 MB), `best.pt` (5.55 MB),
`classes.json` (87 classes), `TRAINING_REPORT.json`, and
`DATASET_SOURCES.txt`. The ZIP integrity/CRC check passed.

## Notebook-reported metrics, not a new independent evaluation

| Metric | Value |
| --- | ---: |
| Requested three stages | 35 + 25 + 15 epochs |
| Selected checkpoint | Refinement (stage 2) |
| Test mAP@50 | 0.6995 |
| Test mAP@50–95 | 0.4750 |
| Class labels | 87 |
| Specialist holdout test images | 9 |

**Known gaps:** Five label names have no test-class AP row (LDR, Sound,
LDR Sensor, Ceramic Capacitor, Resistor 52K). ESP32, Joystick, LED Matrix,
and Stepper Driver have reported AP@50 of zero. The report says
near-duplicate leakage was **not audited**, and that new test set composition
invalidates a direct comparison with old historical test mAP.

Several near-perfect per-class scores have very small sample counts and
must not be interpreted as perfect deployment accuracy. Predicted resistor
labels never mean resistance was electrically measured.

## Next steps before activating on the original website

1. Verify the export performs an actual ONNX forward pass in Google Colab
   and correctly produces 87 class channels.
2. Test several **new** photos against plain and patterned backgrounds, both
   with and without electronics. Pay particular attention to the earlier
   non-component false positive (issue #10: `Soil Moisture`).
3. Compare these same test images with the currently deployed CircuitVision
   upload detector, capturing missed detections and false positives by class.
4. Test live camera motion and render performance after real deployment.
   Tracker-only changes do not fix a true background-classification error.
5. Audit near-duplicates and missing rare-class holdout coverage. Retrain
   or curate more examples if necessary.
6. Only after passing the pre-agreed acceptance checks, stage
   `models/unified_87.onnx` and `models/unified_87_classes.json` in this
   existing GitHub repository, verify Render memory/inference/health,
   and activate `CV_ENABLE_UNIFIED_87=1`. Retain rollback to the old
   production model.

No new binary has been committed or activated. The current deployed website
is unchanged. This document is engineering evidence, not public-facing UI copy.
