"""Optional Sam YOLO11s refinement for CircuitVision's EXISTING website.

The original Round 2/Round 3 detections remain authoritative for the broad
component category. Sam's model may provide a visual subcategory suggestion,
never a measured electrical value. No synthetic confidence scores.
"""
from __future__ import annotations

from copy import deepcopy

from backend.smart_fusion import bbox_iou

SAM_CLASS_NAMES = (
    'BUZZER', 'CERAMIC CAPACITOR', 'DIODE', 'LED (BLUE)', 'LED (GREEN)',
    'POTENTIOMETER', 'POWER MODULE', 'RESISTOR 10K', 'RESISTOR 2K',
    'RESISTOR 52K',
)
DETAIL_NAMES = {
    'CERAMIC CAPACITOR': 'Ceramic Capacitor',
    'LED (BLUE)': 'LED (Blue)',
    'LED (GREEN)': 'LED (Green)',
    'RESISTOR 10K': 'Resistor 10K',
    'RESISTOR 2K': 'Resistor 2K',
    'RESISTOR 52K': 'Resistor 52K',
}
GENERIC_NAMES = {
    'BUZZER': 'Buzzer', 'DIODE': 'Diode',
    'POTENTIOMETER': 'Potentiometer', 'POWER MODULE': 'Power Module',
}


def family(label: str) -> str:
    name = str(label).lower().strip()
    if 'resistor' in name and 'photo' not in name:
        return 'resistor'
    if name.startswith('led') or name.startswith('light emitting diode'):
        return 'led'
    if 'capacitor' in name:
        return 'capacitor'
    if 'power module' in name:
        return 'power module'
    if 'potentiometer' in name:
        return 'potentiometer'
    if 'buzzer' in name:
        return 'buzzer'
    if 'diode' in name:
        return 'diode'
    return name


def valid_detection(d: dict) -> bool:
    try:
        bb = d.get('bbox')
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            return False
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bb):
            return False
        return bb[0] < bb[2] and bb[1] < bb[3] and 0 <= float(d.get('confidence', 0)) <= 1
    except (TypeError, ValueError):
        return False


def refine_with_sam(broad: list[dict], specialist: list[dict],
                    min_detail_conf: float = 0.78,
                    matching_iou: float = 0.45) -> list[dict]:
    """Refine only compatible broad labels; reject competing value guesses.

    Preserve the original detection if models disagree or Sam lacks confident
    evidence. For novel isolated generic objects, require still higher confidence.
    """
    output = deepcopy(broad)
    candidates = [d for d in specialist if valid_detection(d) and
                  str(d.get('label', '')).upper() in SAM_CLASS_NAMES]
    used = set()

    for original in output:
        if not valid_detection(original):
            continue
        possible = [(idx, d) for idx, d in enumerate(candidates)
                    if family(original.get('label', '')) == family(d['label']) and
                    bbox_iou(original['bbox'], d['bbox']) >= matching_iou and
                    float(d['confidence']) >= min_detail_conf]
        details = [(idx, d) for idx, d in possible if d['label'].upper() in DETAIL_NAMES]
        if not details:
            continue
        # Competing specific values/colors for the same part are ambiguous;
        # don't quietly promote the most confident answer to an exact value.
        labels = {d['label'].upper() for _, d in details}
        if len(labels) != 1:
            original['review_status'] = 'specialist_conflict'
            continue
        idx, best = max(details, key=lambda x: float(x[1]['confidence']))
        used.add(idx)
        original['original_label'] = original.get('label')
        original['core_confidence'] = original.get('confidence')
        original['label'] = DETAIL_NAMES[best['label'].upper()]
        original['confidence'] = float(best['confidence'])
        original['source_model'] = 'sam_yolo11s'
        original['review_status'] = 'specialist_visual_estimate'
        original['specialist_label'] = str(best['label'])

    for idx, candidate in enumerate(candidates):
        if idx in used or candidate['label'].upper() not in GENERIC_NAMES:
            continue
        if float(candidate['confidence']) < 0.86:
            continue
        if any(bbox_iou(candidate['bbox'], existing.get('bbox')) >= 0.35
               for existing in output if valid_detection(existing)):
            continue
        fresh = deepcopy(candidate)
        fresh['label'] = GENERIC_NAMES[candidate['label'].upper()]
        fresh['source_model'] = 'sam_yolo11s'
        fresh['review_status'] = 'specialist_candidate'
        output.append(fresh)
    return sorted(output, key=lambda d: float(d.get('confidence', 0)), reverse=True)
