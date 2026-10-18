# Publish CircuitVision as a free HTTPS website (Render)

A public link will be assigned **only after you deploy through your account**. You cannot use the example names below as a verified, running website link.

## Before deployment

1. Train a real YOLO object detector (see `TRAIN_AND_EXPORT_COLAB.md`). You must have YOLO bounding-box annotations, not just class folders.
2. Export **`best.onnx`** from Colab using YOLO11 `nms=False, dynamic=False` (the lightweight backend is designed for raw YOLO11 detection output).
3. Put the file into **`models/best.onnx`** in this project. If the file is unusually large, GitHub may require Git LFS; do not confuse that with the original `.pt` file.
4. **Test locally** with `python -m pip install -r requirements.txt` then `uvicorn backend.main:app --host 127.0.0.1 --port 8000`. Visit `/api/health` and run image detection first. Camera needs HTTPS (on mobile) or localhost.

## Deploy on Render Free (requires your account)

1. Sign into [GitHub](https://github.com/) and create a repository, e.g. `circuitvision-electronics-yolo`.
2. Upload the contents of this folder to the **repository root**, including `render.yaml`, `models/best.onnx`, `index.html`, `app.js`, `styles.css`, and the `backend/` folder.
3. Visit [Render Dashboard](https://dashboard.render.com/). Connect your GitHub account.
4. Choose **New → Blueprint** and select the repository. Approve its `render.yaml`. The service type is **Web Service / Free**. (You can also choose New → Web Service, `pip install -r requirements.txt`, `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`, plan Free.)
5. Wait until Render reports **Live / Deploy succeeded**. Open the real `https://<your-assigned-app>.onrender.com` address and verify `/api/health` returns `custom_model_available: true`.
6. Open the URL on your **phone using Chrome or Safari**, tap Live Camera, grant camera permission, and tap Start YOLO detection. Then test uploading an image, and downloading the annotated image/JSON/CSV. Send **that confirmed HTTPS URL** to your professor.

### Honest free-tier limitations

- Render Free has **512MB memory and 0.1 CPU**. ONNX Runtime avoids the much heavier PyTorch server, but your large YOLO11s ONNX file may still exceed memory/CPU limits. If the Render instance crashes or frame processing is extremely slow, export a smaller retrained `yolo11n.pt` detection model (not generic COCO weights) or use another host with more memory. **No minimum FPS is promised.**
- A free web service spins down after **15 minutes idle**; the first visit after sleep may take **about a minute** to wake up. Open it and verify it loads before your presentation.
- Free Render may enforce bandwidth/build usage limits. Deployment may require account verification and your authorization; this ZIP cannot create a web service on your behalf.
- Webcam permissions require **HTTPS** for other devices. The page sends images/frames to the server for inference but does not store them.
- Public source repositories expose the model file and code. Check the dataset/model licenses before public upload.

## If the website shows "model missing"

It means the hosting is live but no `.onnx` weights have been added to `models/`. A Colab link and Kaggle page are not equivalent to trained model weights. Don't present sample detections as actual experimental results.

## Endpoints

- `GET /` — website
- `GET /api/health` — deployment and trained-weight availability
- `POST /api/detect` — multipart file named `image`, confidence 0.01–0.99, model=`custom`.
- `GET /api/docs` — API schema

Input JPEG, PNG, or WEBP, max 10 MB. Output `detections[{label, class_id, confidence, bbox}]`, where bbox is normalized `[x1,y1,x2,y2]`.
