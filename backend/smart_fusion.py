"""Conservative automatic CircuitVision YOLO11 output fusion.

Uses the student's verified Round 2 as primary. The expanded Round 3 is
allowed to contribute ONLY genuinely additional categories on still photos,
after overlap/confidence checks; it never overrides a core prediction. This
is not model-weight merging or proof of improved accuracy.
"""
from __future__ import annotations

import re
from copy import deepcopy

def canonical(label: str) -> str:
    s = re.sub(r'[^a-z0-9]+', ' ', str(label).lower()).strip()
    s = ' '.join(s.split())
    aliases = {
        '7 segment': '7 seg', '7segment': '7 seg', 'seven segment': '7 seg',
        '9 volt battery': 'battery 9v', '9v battery': 'battery 9v',
        'arduino uno r3': 'arduino uno',
    }
    return aliases.get(s,s)

def bbox_iou(a, b) -> float:
    if not a or not b or len(a)!=4 or len(b)!=4:
        return 0.0
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-8,area_a+area_b-inter)

def auto_fuse(core: list[dict], expanded: list[dict],
              core_names, minimum_extra_confidence: float=0.55) -> list[dict]:
    """Preserve core detections, append supported additional classes only.

    No synthetic or averaged confidence scores. A new class needs its own
    high-confidence Round 3 prediction and must not overlap a core box.
    """
    known={canonical(s) for s in core_names}
    results=[dict(deepcopy(d),source_model='core_yolo11') for d in core]
    for candidate in sorted(expanded,key=lambda x:x.get('confidence',0),reverse=True):
        name=canonical(candidate.get('label',''))
        box=candidate.get('bbox')
        score=float(candidate.get('confidence',0))
        if not name or name in known or score<minimum_extra_confidence:
            continue
        if not box or len(box)!=4 or not all(isinstance(x,(float,int)) and 0<=x<=1 for x in box):
            continue
        if box[0]>=box[2] or box[1]>=box[3]:
            continue
        if any(bbox_iou(box,x.get('bbox'))>=0.38 for x in results):
            continue
        d=deepcopy(candidate)
        d['source_model']='expanded_yolo11'
        d['review_status']='experimental_extra_class'
        results.append(d)
    return sorted(results,key=lambda x:x.get('confidence',0),reverse=True)
