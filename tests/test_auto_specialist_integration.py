"""Contract tests: the existing CircuitVision API actually uses the optional Sam model."""
from PIL import Image

from backend import main

class FakeONNX:
    def __init__(self, kind):
        self.kind = kind
        self.names = {0: 'Resistor'} if kind == 'custom' else {
            0: 'RESISTOR 10K'} if kind == 'sam' else {0: 'Oscilloscope'}

    def predict(self, _image, _conf):
        if self.kind == 'custom':
            return [{'label':'Resistor', 'class_id':0,'confidence':.84,'bbox':[.1,.1,.5,.5]}]
        if self.kind == 'sam':
            return [{'label':'RESISTOR 10K','class_id':0,'confidence':.9,'bbox':[.1,.1,.5,.5]}]
        return [{'label':'Oscilloscope','class_id':0,'confidence':.74,'bbox':[.6,.6,.9,.9]}]

def test_upload_runs_all_available_models_but_camera_uses_fast_core(monkeypatch,tmp_path):
    (tmp_path/'round3.onnx').touch()
    (tmp_path/'sam_10.onnx').touch()
    monkeypatch.setattr(main,'MODEL_FOLDER',tmp_path)
    called=[]
    def fake_get(kind):
        called.append(kind)
        return (FakeONNX(kind),'onnx')
    monkeypatch.setattr(main.store,'get',fake_get)
    image=Image.new('RGB',(80,80),'white')

    live=main.infer(image,'auto',.35,live=True)
    assert called==['custom']
    assert [d['label'] for d in live] == ['Resistor']

    called.clear()
    still=main.infer(image,'auto',.35,live=False)
    assert 'round3' in called and 'sam' in called
    assert {d['label'] for d in still} == {'Resistor 10K','Oscilloscope'}
    assert any(d['review_status']=='specialist_visual_estimate' for d in still)

def test_optional_sam_absence_does_not_break_detection(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'MODEL_FOLDER',tmp_path)
    monkeypatch.setattr(main.store,'get',lambda kind: (FakeONNX(kind),'onnx'))
    result=main.infer(Image.new('RGB',(80,80)), 'auto', .35)
    assert len(result)==1 and result[0]['label']=='Resistor'
