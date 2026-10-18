"""Validate and prepare annotated electronic-component images for YOLO detection.

Accepts either:
  - Label Studio YOLO export: images/, labels/, classes.txt
  - Existing YOLO dataset with image/label pairs and data.yaml.
  - A ZIP containing either layout.

Refuses classification-only folders / unannotated images. No fake bounding
boxes are created. Uses a deterministic 90/10 split (configurable).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def safe_unzip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        root = target.resolve()
        for record in z.infolist():
            extracted = (root / record.filename).resolve()
            if not extracted.is_relative_to(root):
                raise ValueError('Unsafe path in zip archive; refusing extraction.')
        z.extractall(root)


def available_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not any(part.startswith('.') for part in p.relative_to(root).parts))


def find_classes(root: Path) -> list[str]:
    import yaml
    yaml_files = sorted([*root.rglob('data.yaml'), *root.rglob('dataset.yaml')], key=lambda x: len(x.parts))
    for file in yaml_files:
        try:
            meta = yaml.safe_load(file.read_text(encoding='utf-8'))
            names = meta.get('names') if isinstance(meta, dict) else None
            if isinstance(names, list) and names:
                return [str(name) for name in names]
            if isinstance(names, dict) and names:
                keys = sorted(names, key=lambda n: int(n))
                if [int(i) for i in keys] == list(range(len(keys))):
                    return [str(names[k]) for k in keys]
        except (ValueError, TypeError, OSError, KeyError):
            pass
    for file in sorted(root.rglob('classes.txt'), key=lambda x: len(x.parts)):
        names = [line.strip() for line in file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if names:
            return names
    return []


def find_pairs(images: list[Path], root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    labels = [p for p in root.rglob('*.txt') if 'labels' in [q.lower() for q in p.parts]
              and p.name.lower() != 'classes.txt']
    # Some exports store all labels in the same folder. Fail on ambiguous names.
    index: dict[str, Path] = {}
    for label in labels:
        if label.stem in index:
            raise ValueError(f'Duplicate label basename {label.stem!r}: {index[label.stem]} and {label}. '
                             'Rename overlapping image/label pairs before preparation.')
        index[label.stem] = label
    seen: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for image in images:
        if image.stem in seen:
            raise ValueError(f'Duplicate image basename {image.stem!r}. Rename duplicates first.')
        seen.add(image.stem)
        if image.stem in index:
            pairs.append((image, index[image.stem]))
        else:
            missing.append(image)
    return pairs, missing


def validate_labels(pairs: list[tuple[Path, Path]], names: list[str]) -> int:
    total = 0
    for image, label in pairs:
        for i, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), start=1):
            if not line.strip():
                continue
            items = line.split()
            if len(items) != 5:
                raise ValueError(f'{label}:{i}: expected YOLO detection label "class x_center y_center width height"; got {len(items)} fields.')
            try:
                cat = int(items[0]); x, y, w, h = map(float, items[1:])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{label}:{i}: malformed class or box numbers.') from exc
            if not (0 <= cat < len(names)):
                raise ValueError(f'{label}:{i}: class {cat} outside class map 0..{len(names) - 1}.')
            if not all(0 <= q <= 1 for q in [x, y, w, h]) or w <= 0 or h <= 0:
                raise ValueError(f'{label}:{i}: invalid normalized box {x} {y} {w} {h}.')
            if x - w / 2 < -0.005 or x + w / 2 > 1.005 or y - h / 2 < -0.005 or y + h / 2 > 1.005:
                raise ValueError(f'{label}:{i}: bounding box extends outside the image.')
            total += 1
    return total


def prepare(source: Path, output: Path, val_ratio: float = 0.1, seed: int = 42) -> dict:
    import yaml
    if not 0.05 <= val_ratio <= 0.5:
        raise ValueError('Validation split must be between 5% and 50%.')
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f'Source does not exist: {source}')
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError('Output cannot be inside the input dataset folder.')
    with tempfile.TemporaryDirectory(prefix='circuitvision_yolo_') as scratch:
        dataset = source
        if source.is_file():
            if not zipfile.is_zipfile(source):
                raise ValueError('Source must be an extracted folder or .zip dataset.')
            dataset = Path(scratch)
            safe_unzip(source, dataset)
        images = available_images(dataset)
        if not images:
            raise ValueError('No image files found in the dataset.')
        names = find_classes(dataset)
        pairs, missing = find_pairs(images, dataset)
        print(f'Found {len(images)} images, {len(pairs)} image/label pairs, {len(missing)} missing label files.')
        if not pairs:
            raise ValueError('This dataset has images but no YOLO detection bounding-box label files. '
                             'Class folders or filenames are NOT enough for bounding-box training. '
                             'Annotate the images in Label Studio, export YOLO detection labels, and retry.')
        if missing:
            raise ValueError(f'{len(missing)} images lack matching .txt annotations (for example: {missing[0].name}). '
                             'Add an empty .txt label file for verified no-object images, or annotate the missing images.')
        if not names:
            raise ValueError('No classes.txt or data.yaml with class names found. Provide the verified class map from your labels.')
        count = validate_labels(pairs, names)
        if not count:
            raise ValueError('All label files are empty. No bounding boxes exist to train an object detector.')
        if len(pairs) < 2:
            raise ValueError('At least two annotated images are needed for a train/validation split.')
        random.Random(seed).shuffle(pairs)
        val_num = max(1, min(len(pairs) - 1, round(len(pairs) * val_ratio)))
        split = {'val': pairs[:val_num], 'train': pairs[val_num:]}
        output.mkdir(parents=True, exist_ok=True)
        for part, rows in split.items():
            idir = output / 'images' / part
            ldir = output / 'labels' / part
            idir.mkdir(parents=True, exist_ok=True)
            ldir.mkdir(parents=True, exist_ok=True)
            for img, lab in rows:
                shutil.copy2(img, idir / img.name)
                shutil.copy2(lab, ldir / lab.name)
        config = {
            'path': str(output),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(names),
            'names': names,
        }
        (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        report = {'images': len(images), 'boxes': count, 'classes': names,
                  'train': len(split['train']), 'val': len(split['val']),
                  'yaml': str(output / 'data.yaml')}
        print('READY:', report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Kaggle download folder or YOLO Label Studio ZIP')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--val-ratio', type=float, default=.1)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.val_ratio)
    except Exception as exc:
        print('DATASET NOT READY:', exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
