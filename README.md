# CircuitVision — AI-Assisted Electronics Laboratory

A frontend + Python inference service for electronic-component **object detection** using YOLO11. The original student Colab trains YOLO11s; this web app accepts a **trained custom detector** and returns *actual* bounding boxes from the model.

## Features

- Live laptop and mobile camera (HTTPS), with browser-based video and frame-by-frame YOLO predictions.
- Upload a JPG, PNG or WEBP and obtain detected class names, confidence, bounding-box coordinates, and result exports.
- Confidence controls, box and label toggles, model readiness status, proper errors for missing model files.
- Same-origin FastAPI server, so a professor only needs the public website link once it's deployed.
- Lightweight `.onnx` model inference for constrained CPU hosting; local `.pt` works with optional Ultralytics installation.

**Deployment status:** This source package is deployment-ready, but there is **no hosted public URL yet**. No trained model was supplied. Without `models/best.onnx`, real detection correctly reports an error instead of faking results.

## How to finish

1. Read [`TRAIN_AND_EXPORT_COLAB.md`](TRAIN_AND_EXPORT_COLAB.md) and confirm your Kaggle data has YOLO bounding-box annotations.
2. Train and export **best.onnx** using the code provided.
3. Put `best.onnx` in `models/`.
4. Follow [`DEPLOY_FREE.md`](DEPLOY_FREE.md) to publish on Render's free HTTPS address.
5. Verify phone camera, upload, actual detections, and result download *on the deployed site*.

## Local run (ONNX)

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`; the Windows batch file can set up optional `.pt` support with a larger PyTorch install.

## Technical note

The `.onnx` pipeline expects an Ultralytics YOLO11 detection model exported with `dynamic=False` and `nms=False`, which produces class probabilities and XYWH boxes. The backend normalizes labels to the original image size and performs class-aware NMS. The browser draws boxes over the frame and receives no invented class confidences.
