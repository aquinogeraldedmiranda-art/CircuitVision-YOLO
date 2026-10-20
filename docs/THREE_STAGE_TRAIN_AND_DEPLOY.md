# CircuitVision: one website, three training stages, one final YOLO11 model

This document is for the existing [CircuitVision website](https://circuitvision-yolo-lab.onrender.com/).
It is not another web application. The current website remains the baseline until
the proposed unified model passes fair evaluation on the same test images.

## What can and cannot be combined

- Round 2: YOLO11n, 57 classes (production baseline).
- Round 3: YOLO11n, 81 classes (additional lab equipment categories).
- Sam: YOLO11s, 10 classes (LED colors, resistor visual subtypes etc.).
- **You cannot concatenate or average the weights** of models with different
  architectures and different class heads into one valid model.
- The 3 models can coexist in one `/api/detect` inference workflow:
  Round 2 on live frames, Round 2/Round 3/Sam (if installed) on uploaded
  or captured stills. A **genuinely unified model** requires combined, correctly
  mapped annotated data and new training.
- The intended new model uses 87 classes: 81 previous categories + 6
  detailed visual subtypes. Four generic Sam categories map to existing labels.

## Three training stages — in Google Colab

1. Open [the existing 87-class three-stage training notebook](../notebooks/CircuitVision_Unified_Three_Stage_Training.ipynb)
   in Google Colab (File > Open notebook > GitHub).
2. Runtime > Change runtime type > choose a GPU, then run all cells.
3. The notebook checks Sam's **annotated** original `data.zip` (not just
   `my_model.pt`) and downloads the four public Roboflow sets with your
   private API key if they are not already cached.
4. It creates one class-ID map, holds out validation/test images, removes
   byte-identical duplicates, and creates a combined annotated dataset.
5. Three **consecutive** fine-tuning passes are run, each starting from the
   best checkpoint of the previous stage:
   - Foundation: 35 epochs, full augmentation.
   - Refinement: 25 epochs, reduced augmentation.
   - Fine-tuning: 15 epochs, gentler augmentation.
6. The winning checkpoint is selected by validation performance and
   **only then** evaluated against a held-out test set.
7. Export to an ONNX detection model with all 87 class names and a training
   report. Retain the complete report and the original old checkpoints.
   Export files are in `CircuitVision_Unified_YOLO11_Candidate.zip`.

**Three stages are not proof of improved accuracy.** Sam has only 86 distinct
photographs in the previously examined 98-image archive, including 12
duplicated-image groups with disagreeing bounding-box coordinates. Those
conflicts must be reviewed, and nearby augmented images can still leak between
splits. Real phone/lab validation should include motion, clutter, varied
backgrounds, LEDs, resistors, and unseen values. A visual resistor-value class
is not an electrical measurement; confirm with markings or a multimeter.

## Two selectable detectors in the EXISTING website

The user explicitly requested both models rather than replacing the current
working website checkpoint. The public labels are **YOLOv11 (1)** and
**YOLOv11 (2)**.

- YOLOv11 (1): current trained detector, the default for Upload and Camera.
  Its existing model files and inference behavior stay intact.
- YOLOv11 (2): the new 87-class candidate, loaded separately *only if*
  `models/unified_87.onnx` is installed alongside
  `models/unified_87_classes.json`. The model selector is disabled until
  those files are present. A missing model never silently substitutes (1).
- The selected option is shared across Upload and Live Camera in the current
  browser session. Switching models stops the active live detector and
  requires restarting detection, preventing mismatched tracking boxes.
- The health endpoint exposes `yolov11_1_available`,
  `yolov11_2_available` and `yolov11_2_class_count`.

**Installing (2):** upload `models/unified_87.onnx` to the existing GitHub
`main` branch using the original Colab ZIP's `best.onnx` renamed to that
filename. The exact 87-class map is already tracked in this repository.
Render's existing automatic deployment will then incorporate the file.
Do not overwrite `best.onnx`, set `CV_ENABLE_UNIFIED_87`, or create another
Render service.

The candidate was successfully tested on five photos but missed a green LED
and confused a 9V battery with an LCD. It is therefore **experimental**.
Do not describe it as more accurate than the working model. Keep model (1)
as the default. These engineering notes are not public website content.

## Runtime motion/detection safeguards

- Native camera preview is never blocked by YOLO CPU inference.
- Frame-aligned browser optical-flow tracking interpolates boxes only for
  genuine YOLO objects; lost or stale boxes are removed.
- Still-image high-resolution tiling runs only for large uploads, not live.
- Round 3 and Sam network usage is limited on CPU. Sam's model requires a
  verified `models/sam_10.onnx` export before its visual subtypes can appear.
- None of these engineering safeguards are a guarantee of zero bugs or 100%
  detector accuracy; test on actual Windows/Android cameras.
