# CircuitVision — Colab training and model export

Based on your Colab source notebook: `YOLO: You Only Look Once – An AI-Assisted Electronics Laboratory`. Your original settings use **YOLO11s**, 60 epochs, 640-pixel training images and the output `runs/detect/train/weights/best.pt`. We preserve these choices unless you deliberately change them.

**Important:** Your proposed [Kaggle electronic-components dataset](https://www.kaggle.com/datasets/aryaminus/electronic-components) may be a **classification image dataset**, not a YOLO object-detection dataset. We could not directly inspect its full file list. **Do not use an image category folder as a bounding-box label.** The preparation script requires a matching YOLO `labels/*.txt` annotation file per image with actual object coordinates, plus `classes.txt` or `data.yaml`. It refuses unlabeled data.

## Step A — Get the annotated dataset

For a detection model, you need images annotated with a box *around each electronics component*. Acceptable Label Studio export format:

```
custom_data/
  classes.txt       # each class on a separate line, e.g., Power Module
  images/
    0001.jpg
    0002.jpg
  labels/
    0001.txt         # each row: class_id x_center y_center width height, normalized 0–1
    0002.txt
```

If Kaggle includes only images grouped by class (for example `Resistors/`, `Capacitors/`), annotate them with Label Studio or CVAT and export **YOLO detection**, or select a dataset that explicitly includes YOLO bounding boxes. If you need to detect **Power Module**, ensure that exact class is represented and consistently labeled in the annotated training data.

## Step B — In Google Colab, inspect/download the Kaggle dataset

```python
!pip -q install kagglehub pyyaml
import kagglehub
from pathlib import Path
kaggle_root = Path(kagglehub.dataset_download('aryaminus/electronic-components'))
print(kaggle_root)
for p in list(kaggle_root.rglob('*'))[:45]:
    if p.is_file():
        print(p.relative_to(kaggle_root))
```

If there are no image-matched detection `.txt` labels and class names, **stop here** and annotate before training. If you already have an annotated Label Studio ZIP, upload it to Colab instead of using the classification dataset directly.

## Step C — Prepare the YOLO train/validation split

Upload this project's `training/prepare_yolo_dataset.py` into Colab (files sidebar), or copy it from your repository to `/content/prepare_yolo_dataset.py`.

```python
# Set to kaggle_root ONLY if Kaggle includes YOLO detection labels.
annotated_root = kaggle_root
!python /content/prepare_yolo_dataset.py --source "$annotated_root" --output /content/circuitvision_data
```

If the script reports "DATASET NOT READY", fix the annotations; do not try to bypass the check.

## Step D — Train using your YOLO11 code

In a Colab runtime with a GPU enabled, run:

```python
!pip -q install ultralytics onnx
!yolo detect train data=/content/circuitvision_data/data.yaml model=yolo11s.pt epochs=60 imgsz=640 project=/content/circuitvision_runs name=electronics exist_ok=True
```

Check validation mAP50-95, precision, recall and images with overlaid predictions before calling the model accurate. Accuracy is **not** determined by the website layout.

## Step E — Export small, fixed-shape ONNX weights for the free website

```python
from ultralytics import YOLO
from pathlib import Path
model = YOLO('/content/circuitvision_runs/electronics/weights/best.pt')
# Fixed 416x416 CPU model for Render's constrained 512MB free plan.
onnx_path = model.export(format='onnx', imgsz=416, dynamic=False, nms=False,
                         simplify=False, opset=12)
print('ONNX export:', onnx_path)
from google.colab import files
files.download(str(onnx_path))
```

The `best.onnx` file is separate from `best.pt`. You can use the `.pt` version on a local laptop with PyTorch, but **upload the `.onnx` file to the free Render project**. The model embeds its class labels in ONNX metadata. If your export has no class metadata, create `models/classes.json` containing the class list in the exact training order.

**This app does not ship with trained weights. Until you train and supply those weights, it will show the website but cannot perform actual YOLO predictions.**
