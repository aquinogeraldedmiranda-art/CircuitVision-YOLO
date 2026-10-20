# CircuitVision — AI-Assisted Electronics Laboratory

CircuitVision is our team's electronics-component object detection website. We
designed the interface, connected it to a Python FastAPI inference backend, and
trained YOLO11 models in Google Colab. The website is public on Render and
supports uploaded images and a mobile/laptop camera.

## What currently runs

- **Live camera:** the verified student-trained **Round 2 YOLO11n** model
  (57 class labels) with browser-side tracking that moves bounding boxes
  between server detections. Movement can still be inaccurate when objects are
  blurred, occluded or move too quickly.
- **Uploaded and captured photographs:** Round 2 predictions with optional,
  separately marked categories from Round 3 (81 candidate classes).
  Higher-resolution image crops are used to help detect small components on
  large still photos.
- **Sam's model:** a separate 10-class YOLO11s specialist checkpoint.
  The inference integration exists, but its verified ONNX export has not been
  installed in the current Render service. It does **not** automatically read
  resistor values today.
- **Future unified model:** a three-stage Google Colab training notebook
  prepares one 87-class model using the original annotated datasets. It is a
  candidate, **not yet trained or deployed**.

Detections are model predictions, **not** verified electrical measurements.
A 90% detection confidence should not be described in our project paper as
90% model accuracy. We need an independent held-out laboratory test and
measurements of missed objects, false positives and camera latency.

## Website

**Production:** https://circuitvision-yolo-lab.onrender.com/

**Health:** https://circuitvision-yolo-lab.onrender.com/api/health

The public interface presents one CircuitVision YOLO11 detector with Upload
and Live Camera experiences. Both call the same FastAPI endpoint. The Render
CPU deployment uses ONNX Runtime, not the Ultralytics Python prediction loop.

## Keeping our public website current

The About page at [CircuitVision project progress](https://circuitvision-yolo-lab.onrender.com/#/about)
shows reviewed milestones from `project_status.json` alongside the live
model/version information returned by the Render server. Future functional
updates must revise that file in the same pull request; the GitHub workflow
checks this before merging. Colab results are **not** marked complete until
the actual exported model and evaluation report are reviewed.

See [the project-status maintenance guide](docs/KEEP_WEBSITE_IN_SYNC.md).

## Training and deployment

- [Current training and in-place deployment guide](docs/THREE_STAGE_TRAIN_AND_DEPLOY.md)
- [Three-stage Google Colab notebook](notebooks/CircuitVision_Unified_Three_Stage_Training.ipynb)
- [Sam specialist activation steps](docs/SAM_WEBSITE_INTEGRATION.md)
- [Model attribution](MODEL_CREDITS.md)

Training a model multiple times **does not guarantee** improved performance;
select a checkpoint by validation performance and report separate held-out
test results. Keep previous deployed weights for rollback.

## Local setup

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Use http://127.0.0.1:8000/ in your browser. The stored student-trained ZIP
archives under `models/` are installed during app startup. Uploads return
bounding boxes with actual model confidences; no fabricated detections.
