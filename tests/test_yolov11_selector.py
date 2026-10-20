"""The optional second YOLOv11 detector never replaces the tested default."""
import pytest
from fastapi import HTTPException
from PIL import Image
from backend import main

class FakeDetector:
    names={i:f'Class {i}' for i in range(87)}
    def __init__(self, path, labels):
        self.path=path
        self.labels=labels
    def predict(self, rgb, conf):
        return [{'label':'LED Green','confidence':.72,'class_id':83,
                 'bbox':[.1,.1,.5,.5]}]

def test_missing_candidate_fails_explicitly_but_default_is_untouched(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'MODEL_FOLDER',tmp_path)
    monkeypatch.setattr(main.store,'unified',None)
    with pytest.raises(HTTPException) as ex:
        main.store.get('unified')
    assert ex.value.status_code==503
    assert 'not installed' in ex.value.detail
    assert not main.health()['yolov11_2_available']
    assert main.health()['yolov11_1_available'] in [True, False]

def test_candidate_uses_separate_checkpoint(monkeypatch,tmp_path):
    (tmp_path/'unified_87.onnx').write_bytes(b'fake model handled by stub')
    (tmp_path/'unified_87_classes.json').write_text('{}')
    monkeypatch.setattr(main,'MODEL_FOLDER',tmp_path)
    monkeypatch.setattr(main.store,'unified',None)
    monkeypatch.setattr(main,'ONNXDetector',FakeDetector)
    assert main.health()['yolov11_2_available'] is True
    instance,runtime=main.store.get('unified')
    assert runtime=='onnx'
    assert instance.path.name=='unified_87.onnx'
    assert instance.labels.name=='unified_87_classes.json'
    results=main.infer(Image.new('RGB',(80,80)), 'unified', .35,live=True)
    assert results[0]['label']=='LED Green'
    assert results[0]['confidence']==.72

def test_candidate_without_complete_mapping_is_disabled(monkeypatch,tmp_path):
    (tmp_path/'unified_87.onnx').touch()
    monkeypatch.setattr(main,'MODEL_FOLDER',tmp_path)
    assert main.health()['yolov11_2_available'] is False
