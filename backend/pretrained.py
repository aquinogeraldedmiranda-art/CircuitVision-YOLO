"""Install a pinned third-party EESOB electronics detector for the free demo.
Attribution: Yacin Hamdi, https://github.com/yacin-hamdi/yolo-raspberrypi (MIT).
The upstream model is an Ultralytics YOLO26 ONNX detection export.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger('circuitvision.model')
UPSTREAM_COMMIT = '7e14442337b56456f1452251fca3d3428cd6a933'
MODEL_GIT_BLOB_SHA1 = '50e20a12bf05725efd95cd2f5ba2fe36f83a6e63'
MODEL_SIZE = 9_842_560
MODEL_URL = (
    'https://raw.githubusercontent.com/yacin-hamdi/yolo-raspberrypi/'
    f'{UPSTREAM_COMMIT}/eesob_yolo/models/eesob.onnx'
)

def verify_model(data: bytes) -> bool:
    if len(data) != MODEL_SIZE:
        return False
    git_blob = b'blob ' + str(len(data)).encode() + b'\0' + data
    return hashlib.sha1(git_blob).hexdigest() == MODEL_GIT_BLOB_SHA1

def ensure_model(root: Path) -> bool:
    """Fetch real model weights only if the team's own model isn't installed."""
    if os.getenv('CV_AUTO_EESOB', '1') != '1':
        return False
    model_dir = root / 'models'
    if any((model_dir / p).exists() for p in
           ['best.onnx', 'my_model.onnx', 'best.pt', 'my_model.pt']):
        logger.info('Using team model; skipping example EESOB checkpoint')
        return True
    target = model_dir / 'eesob.onnx'
    if target.is_file() and target.stat().st_size == MODEL_SIZE:
        return True
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        req = Request(MODEL_URL, headers={'User-Agent': 'CircuitVision-Education/1.0'})
        with urlopen(req, timeout=45) as response:
            data = response.read(MODEL_SIZE + 1)
        if not verify_model(data):
            logger.error('EESOB model download failed its pinned integrity check')
            return False
        tmp = target.with_suffix('.onnx.tmp')
        tmp.write_bytes(data)
        tmp.replace(target)
        logger.info('Installed verified upstream EESOB electronics checkpoint (%d bytes)',len(data))
        return True
    except (OSError, ValueError) as exc:
        logger.error('Could not download pretrained EESOB model: %s', exc)
        return False
