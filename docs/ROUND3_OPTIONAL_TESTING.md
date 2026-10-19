# Round 3 side-by-side installation

Keep Round 2 as the default and unchanged. Round 3 activates only after the **exact** student-reviewed `CircuitVision_Round3_Candidate.zip` is uploaded to the public repository's `models/` folder.

Upload on Android: https://github.com/aquinogeraldedmiranda-art/CircuitVision-YOLO/upload/main/models

**Required original filename:** `CircuitVision_Round3_Candidate.zip` (not unzipped; not renamed).

Pinned ZIP SHA256: `bb318c2c8afd6da55d6261bbb0feff6ff4458f0e270e8e8c27dba649c81ddffc`. Contents are expected to include `best.onnx`, `classes.json`, `best.pt`, `TRAINING_REPORT.json`, and `DATASET_SOURCES.txt`.

At startup, `install_experimental_round3` installs `round3.onnx` and `round3_classes.json` **alongside** the approved Round 2 `best.onnx` and `classes.json`. The app sends `model=custom` for Round 2 and `model=round3` for Round 3. `/api/health` exposes `experimental_round3_available`. The Round 3 selector remains disabled until its weights are available.

Model switching restarts live detection so moving-box tracking does not mix predictions across models. All upload/camera APIs and zoom settings remain unchanged. The original Round 2 dataset accuracy benchmark is not directly comparable to the expanded Round 3 test set.

On Render free-tier hosting, loading a second 81-class ONNX model may increase RAM and inference latency. If startup or tests fail, keep Round 2 production and diagnose resource limits rather than enabling an unverified model.
