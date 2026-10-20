"""Static test for Sam's CPU-only dataset audit notebook."""
import ast,json
from pathlib import Path
p=Path('notebooks/CircuitVision_Validate_Sam_Data.ipynb')
data=json.loads(p.read_text())
assert data['nbformat']==4
assert len(data['cells'])>=9
sources=[''.join(c['source']) for c in data['cells'] if c['cell_type']=='code']
for n,source in enumerate(sources):
    ast.parse(source,filename=f'cell{n}')
all_code='\n'.join(sources)
for term in ("Sam_Dataset_Audit_Report.zip","image_label_pairs","class_mapping_matches_sam",
             "box_counts_by_id","can_merge_automatically","zipfile.is_zipfile","file_size",
             "files.upload()", "SHA256", "read(label_path)"):
    assert term in all_code, term
assert 'model.train(' not in all_code
print('Sam archive audit notebook syntax valid; no GPU training or model mutation')
