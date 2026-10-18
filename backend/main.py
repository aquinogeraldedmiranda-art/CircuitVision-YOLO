"""CircuitVision HTTPS website + actual model inference on stills and camera frames.

Privacy: does not persist uploaded images or webcam frames. No fake model output.
"""
from __future__ import annotations

import io
import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.onnx_inference import ONNXDetector

ROOT = Path(__file__).resolve().parent.parent
MODEL_FOLDER = ROOT / 'models'
MAX_BYTES = int(os.getenv('CV_MAX_UPLOAD_MB', '10')) * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
CUSTOM_MODEL = os.getenv('CV_MODEL_PATH', '').strip()
ALLOW_GENERIC = os.getenv('CV_ENABLE_GENERIC', '0') == '1'
app = FastAPI(title='CircuitVision YOLO API', version='3.0.0', docs_url='/api/docs', redoc_url=None)


def custom_weight_path() -> Path | None:
    paths = ([Path(CUSTOM_MODEL).expanduser()] if CUSTOM_MODEL
             else [MODEL_FOLDER / 'best.onnx', MODEL_FOLDER / 'my_model.onnx',
                   MODEL_FOLDER / 'best.pt', MODEL_FOLDER / 'my_model.pt'])
    return next((p.resolve() for p in paths if p.is_file()), None)


class ModelStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.path: str | None = None
        self.model: object | None = None
        self.generic: object | None = None

    def get(self, kind: str):
        with self.lock:
            if kind == 'yolo11n':
                if not ALLOW_GENERIC:
                    raise HTTPException(422, 'The generic model is disabled on this deployment.')
                if self.generic is None:
                    try:
                        from ultralytics import YOLO
                        self.generic = YOLO('yolo11n.pt')
                    except Exception as exc:
                        raise HTTPException(503, 'Generic model unavailable; install Ultralytics for local testing.') from exc
                return self.generic, 'pt'
            path = custom_weight_path()
            if path is None:
                raise HTTPException(503, 'Custom YOLO weights are missing. Export best.onnx from Colab and add it to models/.')
            if str(path) != self.path or self.model is None:
                try:
                    if path.suffix.lower() == '.onnx':
                        instance = ONNXDetector(path, MODEL_FOLDER / 'classes.json')
                        runtime = 'onnx'
                    elif path.suffix.lower() == '.pt':
                        from ultralytics import YOLO
                        instance = YOLO(str(path)); runtime = 'pt'
                    else:
                        raise ValueError('Model must be best.onnx or best.pt')
                except Exception as exc:
                    raise HTTPException(503, f'Could not initialize custom model: {exc}') from exc
                self.path = str(path)
                self.model = (instance, runtime)
            return self.model

store = ModelStore()


@app.get('/api/health')
def health():
    path = custom_weight_path()
    can_run = bool(path and (path.suffix.lower() == '.onnx' and importlib.util.find_spec('onnxruntime') is not None
                             or path.suffix.lower() == '.pt' and importlib.util.find_spec('ultralytics') is not None))
    return {'ok': True, 'custom_model_available': can_run,
            'model_file': path.name if path else None,
            'model_format': path.suffix.lstrip('.') if path else None,
            'message': 'Model file ready for first inference' if can_run else
            ('Model file exists but its runtime is missing' if path else
             'Web app online. Add a trained YOLO11 detection model to enable predictions.'),
            'available_models': ['custom'] + (['yolo11n'] if ALLOW_GENERIC else []),
            'live_camera_supported': True}


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            if raw.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise HTTPException(415, 'Use a JPG, PNG, or WEBP photo.')
            if raw.width * raw.height > MAX_PIXELS:
                raise HTTPException(413, 'Image dimensions are too large.')
            return ImageOps.exif_transpose(raw).convert('RGB')
    except HTTPException:
        raise
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, 'Could not decode the image. Use a JPG, PNG or WEBP file.') from exc


def infer(rgb: Image.Image, model: str, conf: float) -> list[dict]:
    instance, runtime = store.get(model)
    if runtime == 'onnx':
        return instance.predict(rgb, conf)
    bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
    with store.lock:
        try:
            result = instance.predict(source=bgr, conf=conf,
                                      imgsz=int(os.getenv('CV_IMAGE_SIZE','640')), verbose=False)[0]
        except Exception as exc:
            raise HTTPException(500, f'YOLO inference failed: {exc}') from exc
    predictions = []
    if result.boxes is None:
        return predictions
    labels = result.names or {}
    for coords, c, klass in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                               result.boxes.cls.cpu().tolist()):
        category = int(klass)
        x1,y1,x2,y2 = coords
        predictions.append({'label': str(labels.get(category,f'Class {category}')),
                            'class_id': category, 'confidence': round(float(c),5),
                            'bbox': [round(max(0,min(1,x1/rgb.width)),6),round(max(0,min(1,y1/rgb.height)),6),
                                     round(max(0,min(1,x2/rgb.width)),6),round(max(0,min(1,y2/rgb.height)),6)]})
    return predictions


@app.post('/api/detect')
async def detect(
    image: Annotated[UploadFile, File(description='Image or camera frame, JPG/PNG/WEBP')],
    confidence: Annotated[float, Form(ge=0.01, le=0.99)] = 0.50,
    model: Annotated[str, Form()] = 'custom',
    show_boxes: Annotated[bool, Form()] = True,
    show_labels: Annotated[bool, Form()] = True,
):
    if model not in {'custom', 'yolo11n'}:
        raise HTTPException(422, 'Invalid model selection')
    if model == 'yolo11n' and not ALLOW_GENERIC:
        raise HTTPException(422, 'Generic model disabled on this deployment')
    data = await image.read(MAX_BYTES + 1)
    await image.close()
    if not data:
        raise HTTPException(400, 'Image is empty')
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f'Image exceeds {MAX_BYTES // (1024*1024)} MB')
    start = time.perf_counter()
    rgb = await run_in_threadpool(decode_image, data)
    detections = await run_in_threadpool(infer, rgb, model, confidence)
    return {'detections': detections, 'model': model,
            'image_width': rgb.width, 'image_height': rgb.height,
            'processing_ms': round((time.perf_counter()-start)*1000,1),
            'annotated_image_url': None}


FRONTEND_FILES = {'index.html', 'app.js', 'api.js', 'demo.js', 'config.js', 'styles.css'}


@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html', media_type='text/html')


@app.get('/assets/favicon.svg')
def favicon():
    return FileResponse(ROOT / 'assets' / 'favicon.svg', media_type='image/svg+xml')


@app.get('/{filename}')
def frontend_asset(filename: str):
    if filename not in FRONTEND_FILES:
        raise HTTPException(404, 'Not found')
    return FileResponse(ROOT / filename)
