# Sam YOLO11s checkpoint — verified source audit

**2026-10-20:** This audit is based on three files uploaded directly by the student: `my_model(1).pt`, `yolo_detector(1).py`, and `YOLO Notebook(1).ipynb`.

## Checkpoint provenance and exact model metadata

- Original PyTorch checkpoint: `my_model(1).pt`, **19,186,330 bytes**, SHA256 `9528dd6c67291abc6c99e22698521b1a3e436ef516dc7239777275ed91c66b5d`.
- PyTorch zip archive includes `best/data.pkl`; pickle inspection found the normal Ultralytics `DetectionModel` and YOLO11-related modules. We **did not execute the checkpoint** during this offline audit.
- The checkpoint training arguments identify `yolo11s.pt`, task `detect`, 60 epochs, 640-pixel training images.
- **Ten actual class IDs**, in order:
  0. BUZZER
  1. CERAMIC CAPACITOR
  2. DIODE
  3. LED (BLUE)
  4. LED (GREEN)
  5. POTENTIOMETER
  6. POWER MODULE
  7. RESISTOR 10K
  8. RESISTOR 2K
  9. RESISTOR 52K

## Notebook validation — critical limitation

The original attached Colab notebook shows **final validation** on only **10 images with 10 annotated objects**. It reports validation precision **0.801**, recall **0.833**, mAP@50 **0.995**, and mAP@50–95 **0.776**, with only **six of the ten classes** appearing in those validation examples. `RESISTOR 10K` had **one annotated validation object** and reported zero recall at the chosen operating threshold despite a high interpolated AP@50. This does **not** validate accurate resistance readings under real-world phone lighting, unseen backgrounds or all ten classes.

**Do not compare Sam's 0.995 validation mAP@50 to Round 2's 0.4987 test mAP@50**: those numbers come from different datasets and sample sizes.

## Source code and dataset requirements

- The supplied `yolo_detector(1).py` is a Windows/desktop OpenCV webcam application, selects an iVCam/DirectShow camera index and loads the `my_model.pt` checkpoint. It is **not** the mobile browser backend and cannot be deployed unchanged to Render.
- The supplied notebook refers to the *separate* `MyDrive/YOLO/dataset/data.zip` with YOLO annotated images. **That dataset was not among the uploaded files**. The notebook itself includes code, text and some outputs, not the full labeled dataset.
- Sam's checkpoint is **YOLO11s**, substantially larger than the existing Round 2 **YOLO11n**. Running both across every live camera frame would likely raise CPU latency and memory pressure on the current small Render instance.

## Safe planned integration

1. Preserve the known-good unified YOLO11 broad detector and browser-side tracking. Keep user-facing model choice hidden.
2. Export Sam's verified `.pt` to ONNX with 10 verified classes; **do not** install a different arbitrary model archive. Use `notebooks/CircuitVision_Export_Sam_Specialist.ipynb` to generate `Sam_Specialist_ONNX.zip`.
3. Validate the exported ONNX on CPU and pin its hashes before any production change.
4. Add a **specialist refinement** only for detected and spatially compatible resistors/LEDs/capacitors. Present value-specific predictions as **unverified visual hints**, never electrical measurements; reject conflicts instead of silently overriding generic detections.
5. Avoid extra specialist inference on each high-FPS video frame. Start with an on-demand close-up / captured frame workflow to protect smooth video.
6. For actual one-model merged training and independent accuracy evaluation, obtain Sam's complete annotated `data.zip` and add an independent varied holdout; do not fabricate unseen class accuracy.
