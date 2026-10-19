"""Static quality checks for the unified Round2+Round3+Sam Colab notebook."""
import ast
import json
from pathlib import Path

NB=Path('notebooks/CircuitVision_Unified_Three_Stage_Training.ipynb')
obj=json.loads(NB.read_text(encoding='utf8'))
assert obj.get('nbformat')==4
assert len(obj['cells'])>=25
codes=[(i,''.join(c['source'])) for i,c in enumerate(obj['cells']) if c['cell_type']=='code']
for index,code in codes:
    executable='\n'.join(line for line in code.splitlines()
                         if not line.lstrip().startswith(('%','!')))
    try:
        ast.parse(executable)
    except SyntaxError as exc:
        raise RuntimeError(f'Python syntax error in notebook cell {index}: {exc}') from exc
all_source='\n'.join(s for _,s in codes)
for required in (
    'SAM_CLASSES=', 'sam_data_archive', 'Sam_Original_Annotated_Dataset.zip',
    'classes.txt', 'SAM_MAP=', 'SAM_SUBCLASSES=', 'assert len(class_names)==87',
    'assert class_names[:57]==old_names', "MODEL_START=MODEL_PT",
    "STAGES=[('foundation',35", "('refinement',25", "('fine_tune',15",
    "split='test'", 'classes.json', 'best.onnx',
    'CircuitVision_Unified_YOLO11_Candidate.zip'
):
    assert required in all_source, f'Missing safety or training stage: {required}'
assert all_source.index('sam_data_archive') < all_source.index('SOURCES=['), (
    'Sam data preflight should occur before downloading other sources')
assert 'install_accuracy_round2' not in all_source
print('Unified notebook: Python syntax and Sam data preflight, 87 classes, 3 stages, frozen test, safe export passed')
