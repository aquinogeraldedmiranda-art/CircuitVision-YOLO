# Sam specialist integration in the existing CircuitVision website

This is NOT another application. CircuitVision continues to serve its original
`index.html`, `app.js`, `backend/main.py` and the same public `/api/detect`.

## How it works

- **Live Camera:** existing verified 57-class Round 2 model, one ONNX forward
  pass per submitted frame, browser-based replay tracking between replies.
  Round 3/Sam are deliberately not added to every live frame to avoid increasing
  CPU latency and making the preview unusable on low-resource hosting.
- **Uploaded or captured image:** Round 2 + optional Round 3 extra categories;
  optional Sam YOLO11s visually refines the broad category when both predictions
  overlap and agree on component family.
- **Conflict policy:** a resistor visual-value hint appears only when the
  specialist reports one unambiguous supported subtype. Competing 2K, 10K,
  52K predictions leave the broad 'Resistor' label and flag the conflict.
  No network output represents an electrical measurement.
- **Missing specialist model:** website, camera, uploads and core results remain
  functional. Health reports `sam_specialist_available: false`.

## REQUIRED to activate Sam on the deployed website

The team has the verified `my_model(1).pt` checkpoint. The separate 10-class
ONNX export is **not yet in this GitHub repository**. That means Sam's model is
NOT active on the published website until this step is done.

1. In Google Colab, open
   [CircuitVision_Export_Sam_Specialist.ipynb](../notebooks/CircuitVision_Export_Sam_Specialist.ipynb).
2. Upload the exact `my_model(1).pt`. The notebook refuses a different SHA256
   and checks the ten class names. No retraining is required for conversion.
3. Run all notebook cells and download `Sam_Specialist_ONNX.zip`.
4. Extract the ZIP and upload `sam_10.onnx` to the EXISTING GitHub repository
   at `models/sam_10.onnx`. Its `sam_classes.json` file is already committed.
   Do not overwrite `models/best.onnx` or `models/round3.onnx`.
5. Let Render redeploy the same existing service. When healthy, confirm
   `/api/health` reports `sam_specialist_available: true`. Upload a close-up
   photo of a supported part to inspect the qualified visual-hint labels.

The `sam_10.pt` checkpoint can also work locally with Ultralytics installed,
but is unsuitable for the current lightweight Render CPU environment. The
published backend expects `sam_10.onnx`.

## Actual 3-stage unified model training

The three `.pt` weights do not become one jointly learned neural network by
concatenating weights. Genuine joint training needs the original annotated
Round 2 and Round 3 datasets, in addition to Sam's recovered 98-image dataset.
Use the existing three-stage Colab training workflow and independent lab-scene
holdout. Three training stages are an experiment, not an automatic accuracy
guarantee. Deploy a resulting one-model ONNX only after comparative validation
on the same real photos and video.

## Validation caveat

Automated API/fusion/motion tests are not real-world model performance tests.
Camera FPS, jitter, per-class recall and false resistor-value identification
still need testing on Android Chrome and Windows cameras.
