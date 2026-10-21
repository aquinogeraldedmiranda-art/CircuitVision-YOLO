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
- **YOLOv11 (2):** the separately trained 87-class ONNX model has been deployed as an optional model selectable in Upload and Live Camera. Preliminary checks identified missed and incorrect classifications; it is not established as more accurate.

Detections are model predictions, **not** verified electrical measurements.
A 90% detection confidence should not be described in our project paper as
90% model accuracy. We need an independent held-out laboratory test and
measurements of missed objects, false positives and camera latency.

## Website

**Production:** https://circuitvision-yolo-lab.onrender.com/

**Health:** https://circuitvision-yolo-lab.onrender.com/api/health

The public interface offers YOLOv11 (1) (default) and YOLOv11 (2) (87-class candidate), each selectable for Upload and Live Camera. Both use the same FastAPI endpoint. The Render
CPU deployment uses ONNX Runtime, not the Ultralytics Python prediction loop.

## Component reference database

The working application includes a relational **SQLite reference catalog**.
The schema is `database/schema.sql` and initial component records are in
`database/component_seed.json`, generated from the original team-written
`component_details.mjs` guide and verified 87-class label mapping.

- `component_categories` stores category-level educational descriptions and
  electrical verification guidance.
- `components` maps visible component names to category records by foreign key.
- `model_registry` and `model_classes` link the unified YOLO classes to
  component records using declared relational constraints.
- `GET /api/components` searches catalog entries; `GET /api/components/{label}`
  retrieves one reference. `POST /api/detect` joins predicted labels to
  database records using the `component_info` field; the Component Guide
  shows the same records with a built-in fallback only when the DB is offline.
- Uploaded images and camera frames are not stored, and the database does not
  invent electronic measurements, datasheet values or confidence/accuracy.

Initialize and check with:
```bash
python -m backend.component_db init
python -m backend.component_db stats
python -m backend.component_db list --query Resistor
python -m backend.component_db upsert --label "Sample Component" --category general
```

The default SQLite path is `/tmp/circuitvision_components.sqlite3` and
can be overridden with `CV_DATABASE_PATH`. Render's default filesystem
is **ephemeral**, so edits made only to the running database do not survive
re-deployments. The tracked seed JSON and schema recreate the catalog on
every server boot. For persistent manual edits, update the seed or attach a
persistent database service; do not claim multi-user data persistence.
No public database write endpoint is exposed.

## Keeping development records accurate

The public website is a clean **CircuitVision · YOLOv11** product experience
with camera, upload, results, component guide and concise About pages.
Internal training stages and model provenance are **not** shown to website
visitors. We retain the detailed audit in `project_status.json` in GitHub.

Every feature or model change must review that ledger in the same pull request.
Only validated model upgrades should change production inference. New Colab
checkpoints require exported weights and independent evaluation first.

See [the internal maintenance guide](docs/KEEP_WEBSITE_IN_SYNC.md).

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
