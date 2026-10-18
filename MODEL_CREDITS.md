# CircuitVision pretrained detector — acknowledgments

The initially deployed detector is an existing open-source **EESOB** electronics component model authored by **Yacin Hamdi**. It is not trained by the CircuitVision team and is not the team's future YOLO11s model.

- Model repository: https://github.com/yacin-hamdi/yolo-raspberrypi
- Dataset repository: https://github.com/yacin-hamdi/EESOB
- License posted by upstream model repository: MIT (Copyright 2026 Yacin Hamdi). The upstream YOLO26 framework may impose additional licensing requirements; check licenses before reuse beyond an educational project.
- Model file: `eesob_yolo/models/eesob.onnx` (~9.8 MB), pinned to commit `7e14442337b56456f1452251fca3d3428cd6a933` and Git blob SHA1 `50e20a12bf05725efd95cd2f5ba2fe36f83a6e63`.

The backend downloads and checks this public ONNX checkpoint on deployment start. This is a trained model (not fake boxes) but detection quality depends on the camera angle, dataset domain, lighting and the object's appearance.

To replace it later with the class's own trained model, add `models/best.onnx` to the deployment (preferred on free CPU hosting) or provide `CV_MODEL_PATH` to an accessible file, and the team model takes priority. New training cannot run on Render's free CPU web service; use a GPU notebook or another compute service.
