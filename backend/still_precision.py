"""Higher-resolution still-image detection, with class-aware duplicate removal.

Only used on explicit image uploads / camera captures. Live video remains
single-pass to preserve interactive latency on Render CPU.
"""
from __future__ import annotations

import os
from copy import deepcopy

from PIL import Image

from backend.smart_fusion import bbox_iou, canonical


def tile_windows(width: int, height: int, min_dimension: int = 1200):
    """At most four 60%-size overlapping crops; no crops for small images."""
    if min(width, height) < min_dimension:
        return []
    cw, ch = round(width * 0.60), round(height * 0.60)
    return [(x, y, cw, ch)
            for y in (0, height - ch)
            for x in (0, width - cw)]


def transform_detection(d: dict, window, full_width: int, full_height: int):
    """Map normalized tile coordinates to normalized original-image space."""
    if len(d.get('bbox', [])) != 4:
        return None
    x, y, cw, ch = window
    x1, y1, x2, y2 = map(float, d['bbox'])
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    # Reject truncated boxes at artificial tile edges (not at image edges).
    eps = 0.008
    if ((x > 0 and x1 < eps) or
        (y > 0 and y1 < eps) or
        (x + cw < full_width and x2 > 1-eps) or
        (y + ch < full_height and y2 > 1-eps)):
        return None
    result = deepcopy(d)
    result['bbox'] = [(x+x1*cw)/full_width, (y+y1*ch)/full_height,
                      (x+x2*cw)/full_width, (y+y2*ch)/full_height]
    result['detection_view'] = 'detail_crop'
    return result


def deduplicate_predictions(detections: list[dict], iou_threshold: float = .52):
    """Keep genuine confidence from the source model, never fabricate scores."""
    kept = []
    for d in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        if any(canonical(existing.get('label', '')) == canonical(d.get('label', ''))
               and bbox_iou(existing.get('bbox'), d.get('bbox')) >= iou_threshold
               for existing in kept):
            continue
        kept.append(d)
    return kept[:120]


def detect_still_with_tiles(detector, image: Image.Image, confidence: float,
                            initial: list[dict] | None = None):
    """Opt-in-on by default; 4 extra CPU passes only on sufficiently big uploads."""
    full = initial if initial is not None else detector.predict(image, confidence)
    if os.environ.get('CV_STILL_TILES', '1') != '1':
        return full
    windows = tile_windows(image.width, image.height)
    if not windows:
        return full
    collected = list(full)
    for window in windows:
        x, y, cw, ch = window
        crop = image.crop((x, y, x+cw, y+ch))
        for detection in detector.predict(crop, max(confidence, .35)):
            mapped = transform_detection(detection, window,
                                         image.width, image.height)
            if mapped is not None:
                collected.append(mapped)
    return deduplicate_predictions(collected)
