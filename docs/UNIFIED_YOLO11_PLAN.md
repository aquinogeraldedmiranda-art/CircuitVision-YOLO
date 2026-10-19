# CircuitVision YOLO11 — single public detection workflow

**Update date: 20 October 2026.**

Users see one **CircuitVision YOLO11** detector. The website no longer asks users to choose Round 2 versus Round 3. Its two inference paths share the same labels/results contract:

- **Live Android camera:** verified student-trained Round 2 YOLO11n as the core model, with existing smooth native video and browser motion tracking. Do not run multiple slow ONNX networks on every frame merely to increase nominal class coverage.
- **Still image uploads:** Round 2 core predictions, plus selected **additional** Round 3 categories when that verified ONNX is available. The automatic combiner keeps original Round 2 boxes and IDs, rejects colliding expanded predictions, enforces confidence >=55% for experimental new classes, and retains source provenance. It does **not** artificially increase confidence or silently replace a core prediction.
- **Component information:** the component guide and manual 4/5-band resistor calculator remain available. **Resistor ohms are not determined automatically by these two networks.**
- **Internal evaluation:** original checkpoints remain pinned, versioned and recoverable. Replaced/combined weight files were **not** created; merged result lists are not merged model weights.
- **API:** `model=auto` and `live=true|false` for the browser, with legacy `custom` and `round3` API choices retained temporarily for developer regression testing only. Public UI shows one detector.
- **Sam specialist:** Sam's separate YOLO11s 10-class checkpoint and its runtime ensemble source have been recovered from the user's prior project archive. We have NOT installed that PyTorch checkpoint into the free Render CPU service because it would add CPU/RAM overhead and it hasn't passed identical-image validation. The source labels include Resistor 2K, 10K, 52K; LED blue/green; ceramic capacitor; and four generic categories. A specialist classification label is **not an electrical measurement**.

## Next verified unified model (training required)

To make all recognition run inside **one trained checkpoint**, we need actual combined annotated photographs (not simply mixing trained .pt files). Sources already available in Colab:
1. EESOB v4 and Electronics Components SVECk v7 (Round 2).
2. Expanded electronic component v3 and lab equipment v2 (Round 3).
3. Sam's original `MyDrive/YOLO/dataset/data.zip` Label Studio YOLO annotation export. This *has not* been recovered into the current app repository. Request this if the user still has access; the known checkpoint `my_model.pt` alone is insufficient.
4. A consistent taxonomy: 81 general candidate classes plus six detail categories only where reliable; merge synonyms (7 Seg / 7-segment), handle generic vs specific resistor and LED boxes consistently. Keep detail annotation provenance to avoid misleading labels.
5. Freeze a holdout that includes independent smartphone images and moving video. Compare recall/precision/AP per class, false detailed values, localization jitter, inference latency and end-to-end live performance. Keep the current stable production model until a successor proves better.

There is no truthful guarantee of **zero** bugs or **100%** accuracy. Report measured errors and specific untested scenarios instead.
