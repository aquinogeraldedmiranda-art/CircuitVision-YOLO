"""CircuitVision HTTPS website + actual model inference on stills and camera frames.

Privacy: does not persist uploaded images or webcam frames. No fake model output.
"""
from __future__ import annotations

import io
import importlib.util
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from backend import component_db
from backend.onnx_inference import ONNXDetector
from backend.smart_fusion import auto_fuse
from backend.still_precision import detect_still_with_tiles
from backend.specialist_fusion import SAM_CLASS_NAMES, refine_with_sam
from backend.pretrained import ensure_model
from backend.model_archive import install_uploaded_checkpoint, install_accuracy_round2, install_experimental_round3, MODEL_SHA256

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
                   MODEL_FOLDER / 'best.pt', MODEL_FOLDER / 'my_model.pt',
                   MODEL_FOLDER / 'eesob.onnx'])
    return next((p.resolve() for p in paths if p.is_file()), None)


class ModelStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.path: str | None = None
        self.model: object | None = None
        self.generic: object | None = None
        self.round3: tuple | None = None
        self.specialist: tuple | None = None
        self.unified: tuple | None = None

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
            if kind == 'unified':
                # Kept strictly separate from the working website checkpoint.
                # A missing or invalid candidate never falls back silently.
                unified_path = MODEL_FOLDER / 'unified_87.onnx'
                unified_map = MODEL_FOLDER / 'unified_87_classes.json'
                if not unified_path.is_file() or not unified_map.is_file():
                    raise HTTPException(503, 'YOLOv11 (2) is not installed yet. Choose YOLOv11 (1) for now.')
                if self.unified is None:
                    try:
                        instance2 = ONNXDetector(unified_path, unified_map)
                        if len(instance2.names) != 87 or set(instance2.names) != set(range(87)):
                            raise ValueError('YOLOv11 (2) class mapping must contain all 87 labels.')
                        self.unified = (instance2, 'onnx')
                    except Exception as exc:
                        raise HTTPException(503, f'YOLOv11 (2) could not load: {exc}') from exc
                return self.unified
            if kind == 'sam':
                path_sam = MODEL_FOLDER / 'sam_10.onnx'
                pt_sam = MODEL_FOLDER / 'sam_10.pt'
                labels_sam = MODEL_FOLDER / 'sam_classes.json'
                if not labels_sam.is_file():
                    raise HTTPException(503, 'Sam specialist class mapping is missing.')
                if not path_sam.is_file() and not pt_sam.is_file():
                    raise HTTPException(503, 'Sam specialist not installed. Export sam_10.onnx from my_model.pt for web hosting.')
                if self.specialist is None:
                    try:
                        if path_sam.is_file():
                            instance_sam = ONNXDetector(path_sam, labels_sam)
                            runtime_sam = 'onnx'
                        else:
                            from ultralytics import YOLO
                            instance_sam = YOLO(str(pt_sam))
                            runtime_sam = 'pt'
                        if [instance_sam.names[i] for i in range(10)] != list(SAM_CLASS_NAMES):
                            raise ValueError('Specialist names do not match verified 10-class checkpoint.')
                        self.specialist = (instance_sam, runtime_sam)
                    except Exception as exc:
                        raise HTTPException(503, f'Cannot load Sam specialist: {exc}') from exc
                return self.specialist
            if kind == 'round3':
                path3 = MODEL_FOLDER / 'round3.onnx'
                labels3 = MODEL_FOLDER / 'round3_classes.json'
                if not path3.is_file() or not labels3.is_file():
                    raise HTTPException(503, 'Round 3 experimental model is not uploaded yet. Select Round 2 for now.')
                if self.round3 is None:
                    try:
                        instance3 = ONNXDetector(path3, labels3)
                        if len(instance3.names) != 81:
                            raise ValueError('Round 3 must have 81 classes')
                        self.round3 = (instance3, 'onnx')
                    except Exception as exc:
                        raise HTTPException(503, f'Cannot load Round 3: {exc}') from exc
                return self.round3
            path = custom_weight_path()
            if path is None:
                raise HTTPException(503, 'Custom YOLO weights are missing. Export best.onnx from Colab and add it to models/.')
            if str(path) != self.path or self.model is None:
                try:
                    if path.suffix.lower() == '.onnx':
                        instance = ONNXDetector(path, MODEL_FOLDER / ('unified_87_classes.json' if path.name == 'unified_87.onnx' else 'classes.json'))
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


@app.on_event('startup')
async def download_public_lab_checkpoint():
    await run_in_threadpool(component_db.initialize_database)
    # Render's free hosting keeps only build/runtime files, so retrieve the
    # verified pinned public model when the container starts.
    # Prefer the stronger tested Round-2 detector when its verified ZIP has
    # been uploaded. Otherwise keep the first student checkpoint operational.
    installed_student_model = await run_in_threadpool(install_accuracy_round2, ROOT)
    if not installed_student_model:
        installed_student_model = await run_in_threadpool(install_uploaded_checkpoint, ROOT)
    if not installed_student_model:
        await run_in_threadpool(ensure_model, ROOT)
    # Experimental weights use independent paths; never replace Round 2.
    await run_in_threadpool(install_experimental_round3, ROOT)


@app.get('/api/health')
def health():
    try:
        db_state = component_db.database_status()
    except (OSError,sqlite3.Error,ValueError) as exc:
        db_state = {"ready":False,"error":type(exc).__name__}
    path = custom_weight_path()
    can_run = False
    message = 'No trained detector available'
    detected_labels = 0
    if path:
        try:
            instance, runtime = store.get('custom')
            detected_labels = len(instance.names) if runtime == 'onnx' else len(instance.names)
            can_run = True
            message = ('EESOB electronic components model ready' if path.name == 'eesob.onnx' else
                       'CircuitVision student-trained YOLO11n model ready')
        except HTTPException as exc:
            message = f'Model loading failed: {exc.detail}'
    round3_ready = (MODEL_FOLDER / 'round3.onnx').is_file() and (MODEL_FOLDER / 'round3_classes.json').is_file()
    # Do not expose (2) as selectable until its actual ONNX is installed.
    unified_candidate_installed = ((MODEL_FOLDER / 'unified_87.onnx').is_file() and
                                   (MODEL_FOLDER / 'unified_87_classes.json').is_file())
    specialist_ready = (MODEL_FOLDER / 'sam_10.onnx').is_file() or (
        (MODEL_FOLDER / 'sam_10.pt').is_file() and importlib.util.find_spec('ultralytics') is not None)
    specialist_ready = specialist_ready and (MODEL_FOLDER / 'sam_classes.json').is_file()
    return {'ok': True, 'custom_model_available': can_run,
            'database_ready':db_state['ready'],
            'database_component_count':db_state.get('components',0),
            'database_model_2_class_links':db_state.get('mapped_model_2_classes',0),
            'deployment_commit': os.getenv('RENDER_GIT_COMMIT', 'local')[:12],
            'still_image_tiling': os.getenv('CV_STILL_TILES', '1') == '1',
            'trained_single_model_active': bool(path and path.name == 'unified_87.onnx'),
            'unified_yolo11_available': can_run,
            'yolov11_1_available': can_run,
            'yolov11_2_available': unified_candidate_installed,
            'yolov11_2_class_count': 87 if unified_candidate_installed else None,
            'unified_video_strategy': ('single_trained_unified_checkpoint' if path and path.name == 'unified_87.onnx' else 'verified_core_yolo11_with_browser_tracking'),
            'unified_still_strategy': 'core_plus_filtered_new_classes_when_available',
            'specialist_value_model_available': specialist_ready,
            'sam_specialist_available': specialist_ready,
            'sam_refinement_on_live_frames': False,
            'model_file': path.name if path else None,
            'model_format': path.suffix.lstrip('.') if path else None,
            'model_origin': ('Yacin Hamdi / EESOB pretrained model (2026)' if path and path.name == 'eesob.onnx' else
                             'CircuitVision student-trained combined dataset YOLO11n' if path else None),
            'model_version': ('circuitvision-unified-87-yolo11' if path and path.name == 'unified_87.onnx' else
                              'circuitvision-accuracy-round2-57-yolo11n'
                              if path and path.name == 'best.onnx' and (MODEL_FOLDER / 'CircuitVision_Accuracy_Round2.zip').is_file()
                              else 'circuitvision-combined-57-yolo11n'
                              if path and path.name == 'best.onnx' else None),
            'class_count': detected_labels,
            'message': message,
            'available_models': ['auto', 'custom'] + (['unified'] if unified_candidate_installed else []) + (['round3'] if round3_ready else []) + (['sam'] if specialist_ready else []) + (['yolo11n'] if ALLOW_GENERIC else []),
            'experimental_round3_available': round3_ready,
            'experimental_round3_class_count': 81 if round3_ready else None,
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
    except Image.DecompressionBombError as exc:
        # Phone/gallery source images can be larger than the safe pixel limit,
        # even when the compressed PNG is only a few hundred kilobytes.
        raise HTTPException(413, 'This image is too large to analyze. Reduce its dimensions or take a smaller photo.') from exc
    except (UnidentifiedImageError, ValueError, OSError) as exc:
        raise HTTPException(415, 'This image could not be read. Try a new photo or re-save it as a JPG or PNG.') from exc


def infer(rgb: Image.Image, model: str, conf: float, live: bool = False) -> list[dict]:
    if model == 'auto':
        # Live inference stays on the fast, stable Round 2 path. Browser motion
        # tracking interpolates boxes between genuine server predictions.
        primary = infer(rgb, 'custom', conf, live=live)
        if live:
            return primary
        combined = primary
        if custom_weight_path() and custom_weight_path().name == 'unified_87.onnx':
            return combined  # One validated checkpoint supersedes the ensemble.
        # Extra Round 3 classes are optional, never required for core detection.
        if (MODEL_FOLDER / 'round3.onnx').is_file():
            try:
                extra = infer(rgb, 'round3', max(.55, conf))
                baseline, _runtime = store.get('custom')
                combined = auto_fuse(primary, extra, baseline.names.values(),
                                     minimum_extra_confidence=max(.55, conf))
            except Exception:
                pass  # Keep trained core available if optional expansion fails.
        # Sam provides visual hints for select resistor values / LED colors on
        # uploads and captured stills. Do not run it on every live video frame.
        if ((MODEL_FOLDER / 'sam_10.onnx').is_file() or
                (MODEL_FOLDER / 'sam_10.pt').is_file()):
            try:
                specialist = infer(rgb, 'sam', max(.60, conf))
                combined = refine_with_sam(combined, specialist)
            except Exception:
                pass  # A missing/invalid specialist must not break the website.
        return combined
    instance, runtime = store.get(model)
    if runtime == 'onnx':
        if model == 'custom' and not live:
            return detect_still_with_tiles(instance, rgb, conf)
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


@app.get('/api/components')
def components(query: str = '', limit: int = 100):
    if len(query)>120:
        raise HTTPException(422, 'Component search is too long')
    try:
        items=component_db.list_components(query,limit)
    except (sqlite3.Error,OSError,ValueError) as exc:
        raise HTTPException(503, 'Component database is temporarily unavailable') from exc
    return {'source':'sqlite','count':len(items),'items':items}


@app.get('/api/components/{label}')
def component_info(label: str):
    if not label or len(label)>120:
        raise HTTPException(422, 'Invalid component label')
    try:
        info=component_db.get_component(label)
    except (sqlite3.Error,OSError,ValueError) as exc:
        raise HTTPException(503,'Component database is temporarily unavailable') from exc
    if info is None:
        raise HTTPException(404,'Component not found in the current catalog')
    return info


@app.post('/api/detect')
async def detect(
    image: Annotated[UploadFile, File(description='Image or camera frame, JPG/PNG/WEBP')],
    confidence: Annotated[float, Form(ge=0.01, le=0.99)] = 0.50,
    model: Annotated[str, Form()] = 'custom',
    show_boxes: Annotated[bool, Form()] = True,
    show_labels: Annotated[bool, Form()] = True,
    live: Annotated[bool, Form()] = False,
):
    if model not in {'auto', 'custom', 'unified', 'round3', 'sam', 'yolo11n'}:
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
    detections = await run_in_threadpool(infer, rgb, model, confidence, live)
    # Each authentic YOLO prediction is joined to the relational component
    # catalog by class label. No uploaded photo or camera frame is stored.
    reference_status='available'
    try:
        refs=await run_in_threadpool(component_db.lookup_components,
                                     [item.get('label','') for item in detections])
        detections=[{**item,
                     'component_info':refs.get(str(item.get('label','')).casefold())}
                     for item in detections]
    except (sqlite3.Error,OSError,ValueError):
        # Do not break camera inference when reference storage is unhealthy.
        reference_status='unavailable'
        detections=[{**item,'component_info':None} for item in detections]
    return {'detections': detections,
            'reference_status':reference_status, 'model': ('YOLOv11 (2)' if model == 'unified' else 'YOLOv11 (1)' if model == 'auto' else model),
            'inference_strategy': ('low_latency_core' if live else 'core_with_optional_round3_and_sam') if model == 'auto' else model,
            'image_width': rgb.width, 'image_height': rgb.height,
            'processing_ms': round((time.perf_counter()-start)*1000,1),
            'annotated_image_url': None}


NO_STORE_HEADERS = {'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                    'Pragma': 'no-cache', 'Expires': '0'}


def frontend_revision() -> str:
    # Render provides this commit at build/deploy time. Local copies use
    # the app file's timestamp so static assets change after each edit.
    commit = os.getenv('RENDER_GIT_COMMIT', '')
    if len(commit) >= 12 and all(c in '0123456789abcdef' for c in commit.lower()):
        return commit[:12].lower()
    return str((ROOT / 'app.js').stat().st_mtime_ns)


FRONTEND_FILES = {'index.html', 'app.js', 'api.js', 'demo.js', 'config.js', 'styles.css', 'motion_tracker.mjs', 'component_details.mjs', 'robots.txt', 'sitemap.xml'}


@app.get('/')
def home():
    # HTML and bundled modules must not be reused across model/UI releases.
    # Versioned asset URLs also bypass older Chrome heuristic caches.
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    html = html.replace('__CV_ASSET_VERSION__', frontend_revision())
    return HTMLResponse(html, headers=NO_STORE_HEADERS)


@app.get('/assets/favicon.svg')
def favicon():
    return FileResponse(ROOT / 'assets' / 'favicon.svg', media_type='image/svg+xml', headers=NO_STORE_HEADERS)


@app.get('/{filename}')
def frontend_asset(filename: str):
    if filename not in FRONTEND_FILES:
        raise HTTPException(404, 'Not found')
    if filename == 'index.html':
        return home()
    return FileResponse(ROOT / filename,
                        media_type='text/javascript' if filename.endswith('.mjs') else None,
                        headers=NO_STORE_HEADERS)
