"""Minimal CPU-only YOLO11 ONNX postprocessing for low-memory web hosting.

ONNX export MUST use Ultralytics detect model with nms=False, dynamic=False.
Input layout is RGB NCHW; model output is raw XYWH + class probabilities.
"""
from __future__ import annotations

import ast
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image


def parse_class_map(value: str | None) -> dict[int, str]:
    if not value:
        return {}
    try:
        mapping = json.loads(value)
    except (ValueError, TypeError):
        try:
            mapping = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return {}
    if isinstance(mapping, list):
        return {i: str(name) for i, name in enumerate(mapping)}
    if isinstance(mapping, dict):
        try:
            return {int(k): str(v) for k, v in mapping.items()}
        except (ValueError, TypeError):
            return {}
    return {}


def iou_one_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    lower_right = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0, lower_right - top_left)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max(0., box[2] - box[0]) * max(0., box[3] - box[1])
    area_b = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-8)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                        threshold: float = 0.5, maximum: int = 80) -> np.ndarray:
    """Class-aware NMS; sorted indices of retained predictions."""
    order = np.argsort(scores)[::-1]
    retained: list[int] = []
    while len(order) and len(retained) < maximum:
        k = int(order[0]); retained.append(k)
        remaining = order[1:]
        if not len(remaining):
            break
        same = classes[remaining] == classes[k]
        overlap = iou_one_many(boxes[k], boxes[remaining])
        order = remaining[~(same & (overlap > threshold))]
    return np.asarray(retained, dtype=np.int64)


def make_letterbox(image: Image.Image, input_w: int, input_h: int):
    w, h = image.size
    factor = min(input_w / w, input_h / h)
    target_w, target_h = max(1, round(w * factor)), max(1, round(h * factor))
    left = (input_w - target_w) // 2
    top = (input_h - target_h) // 2
    rgb = Image.new('RGB', (input_w, input_h), (114, 114, 114))
    rgb.paste(image.resize((target_w, target_h), Image.Resampling.BILINEAR), (left, top))
    # Return actual x/y scaling separately after pixel rounding.
    arr = np.asarray(rgb, dtype=np.float32)
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None] / 255.0)
    return tensor, (left, top, target_w / w, target_h / h)


def decode_yolo11(output: np.ndarray, meta: tuple[int, int, float, float],
                  image_w: int, image_h: int, class_names: dict[int, str],
                  confidence: float) -> list[dict]:
    """Decode ordinary Ultralytics YOLO11 detection export (no built-in NMS)."""
    output = np.asarray(output)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError('Expected batch size 1 in ONNX output')
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f'Unsupported ONNX output shape {output.shape}. Re-export with nms=False.')
    # Typical YOLO11: 1 x (4+classes) x anchors. Some versions transpose axes.
    if output.shape[0] < output.shape[1]:
        output = output.T
    rows, cols = output.shape
    if cols < 5 or cols > 1000:
        raise ValueError(f'Unexpected YOLO11 detection output {output.shape}. Export task=detect, nms=False.')
    classes = output[:, 4:]
    class_id = np.argmax(classes, axis=1)
    score = classes[np.arange(rows), class_id]
    selected = np.where(np.isfinite(score) & (score >= confidence))[0]
    if not len(selected):
        return []
    # pre-limit memory and NMS CPU work
    selected = selected[np.argsort(score[selected])[::-1][:450]]
    xywh = output[selected, :4].astype(np.float32)
    cls = class_id[selected].astype(np.int32)
    scr = score[selected].astype(np.float32)
    left, top, scale_x, scale_y = meta
    cx, cy, bw, bh = xywh.T
    b = np.stack(((cx - bw / 2 - left) / scale_x,
                  (cy - bh / 2 - top) / scale_y,
                  (cx + bw / 2 - left) / scale_x,
                  (cy + bh / 2 - top) / scale_y), axis=1)
    b[:, 0::2] = b[:, 0::2].clip(0, image_w)
    b[:, 1::2] = b[:, 1::2].clip(0, image_h)
    keep = non_max_suppression(b, scr, cls)
    results = []
    for k in keep:
        x1, y1, x2, y2 = b[k]
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue
        category = int(cls[k])
        results.append({'label': class_names.get(category, f'Class {category}'),
                        'class_id': category,
                        'confidence': round(float(scr[k]), 5),
                        'bbox': [round(float(x1/image_w), 6), round(float(y1/image_h), 6),
                                 round(float(x2/image_w), 6), round(float(y2/image_h), 6)]})
    return results


class ONNXDetector:
    def __init__(self, weight: Path, labels: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('ONNX Runtime is not installed. Install requirements.txt.') from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv('CV_ONNX_THREADS', '1'))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(weight), sess_options=opts, providers=['CPUExecutionProvider'])
        self.lock = threading.Lock()
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError('Expected one input tensor. Export YOLO with nms=False, dynamic=False.')
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f'Expected fixed NCHW [1,3,H,W] image input, got {shape}.')
        self.height, self.width = shape[2], shape[3]
        class_map = self.session.get_modelmeta().custom_metadata_map or {}
        self.names = parse_class_map(class_map.get('names'))
        if labels.exists():
            names_from_file = parse_class_map(labels.read_text('utf-8'))
            if names_from_file:
                self.names = names_from_file

    def predict(self, img: Image.Image, confidence: float) -> list[dict]:
        tensor, meta = make_letterbox(img, self.width, self.height)
        with self.lock:
            raw = self.session.run(None, {self.input_name: tensor})
        if not raw:
            raise ValueError('ONNX model returned no outputs')
        return decode_yolo11(raw[0], meta, img.width, img.height, self.names, confidence)
