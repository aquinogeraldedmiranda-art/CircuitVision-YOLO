"""Install the exact student-trained CircuitVision YOLO11 checkpoint.

User-trained archive is uploaded to models/CircuitVision_Trained_Model.zip.
Pin hashes to the ZIP actually supplied by the student; do not accept substitute
weights or extracts. Keep the zip under the GitHub web upload size limit.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

ARCHIVE_NAME = 'CircuitVision_Trained_Model.zip'
ARCHIVE_SHA256 = '89e601fb299342bfedca6c48addfb04eaaf50ac5831d56c602a85f2cbc4b6d1e'
FILES_SHA256 = {
    'best.onnx': '92dda1dec81ff01a21d23a2918f30472de4d6202aaaee6af6a0011b43b505a5a',
    'classes.json': 'e1d8decbd7ef97f21ecab9e053b9e69de829d8e37df01f43116270b8b2d29503',
}
EXPECTED_MEMBERS = {'best.pt', 'best.onnx', 'classes.json', 'DATASET_SOURCES.txt'}
PRIORITY_LABELS = {'Resistor', 'LED', 'LDR', 'Power Module'}
MODEL_SHA256 = FILES_SHA256['best.onnx']


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install_uploaded_checkpoint(root: Path) -> bool:
    """Return False if archive hasn't been uploaded yet. Fail deployment if invalid."""
    model_dir = root / 'models'
    archive = model_dir / ARCHIVE_NAME
    if not archive.is_file():
        return False
    # 14.3MB archive and 16.3MB total members; cap for zip bombs.
    if archive.stat().st_size > 18_000_000:
        raise RuntimeError('CircuitVision training archive exceeds expected size')
    if sha256(archive.read_bytes()) != ARCHIVE_SHA256:
        raise RuntimeError('CircuitVision training archive SHA-256 mismatch: refuse unverified model')
    with zipfile.ZipFile(archive, 'r') as z:
        if set(z.namelist()) != EXPECTED_MEMBERS or len(z.infolist()) != len(EXPECTED_MEMBERS):
            raise RuntimeError('Unexpected archive files; expected original Colab export')
        if sum(item.file_size for item in z.infolist()) > 19_000_000:
            raise RuntimeError('Uncompressed archive exceeds maximum size')
        verified = {}
        for name, expected_hash in FILES_SHA256.items():
            data = z.read(name)
            if sha256(data) != expected_hash:
                raise RuntimeError(f'Invalid {name} SHA-256 in student-trained archive')
            verified[name] = data
    names = json.loads(verified['classes.json'])
    if (not isinstance(names, dict) or len(names) != 57 or
            set(int(k) for k in names) != set(range(57)) or
            not PRIORITY_LABELS.issubset(set(names.values()))):
        raise RuntimeError('Unexpected model class mapping: expected 57 classes including priority lab parts')
    for name, data in verified.items():
        target = model_dir / name
        if target.is_file() and sha256(target.read_bytes()) == FILES_SHA256[name]:
            continue
        staging = model_dir / (name + '.installing')
        staging.write_bytes(data)
        os.replace(staging, target)
    return True
