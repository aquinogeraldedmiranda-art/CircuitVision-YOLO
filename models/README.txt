ADD YOUR REAL TRAINED MODEL HERE:
  best.onnx       REQUIRED FOR FREE RENDER HOSTING (export from Colab)
  classes.json    optional if ONNX metadata already includes class names

For local use only, best.pt or my_model.pt is also supported when you install
requirements-local-pt.txt. Render uses the lightweight ONNX backend.

No model is bundled. Until you supply real weights, the UI loads but
model-powered detection correctly returns HTTP 503 (not fake results).
