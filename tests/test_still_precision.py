from PIL import Image

from backend.still_precision import tile_windows, transform_detection, deduplicate_predictions, detect_still_with_tiles

def box(label, bb, conf=.9):
    return {'label':label,'bbox':list(bb),'confidence':conf,'class_id':0}

def test_tile_windows_only_for_high_res():
    assert tile_windows(640,480)==[]
    assert len(tile_windows(1600,1200))==4

def test_tile_to_full_frame_placement():
    d=transform_detection(box('Resistor',(.5,.5,.8,.8)),(0,0,600,600),1000,1000)
    assert d['bbox']==[.3,.3,.48,.48]

def test_partial_detections_at_internal_edge_rejected():
    assert transform_detection(box('LED',(0,.2,.2,.8)),(400,0,600,600),1000,1000) is None

def test_class_aware_nms_keeps_distinct_parts():
    xs=[box('Resistor',(.1,.1,.4,.4),.9),
        box('Resistor',(.101,.1,.401,.401),.8),
        box('LED',(.1,.1,.4,.4),.7),
        box('Resistor',(.7,.7,.9,.9),.8)]
    out=deduplicate_predictions(xs)
    assert len(out)==3 and max(x['confidence'] for x in out)==.9

def test_photo_tiling_does_not_change_original_detection(monkeypatch):
    class Mock:
        calls=0
        def predict(self, img, conf):
            self.calls+=1
            return [box('Resistor',(.3,.3,.5,.5),.81)] if self.calls==1 else []
    instance=Mock()
    results=detect_still_with_tiles(instance,Image.new('RGB',(1600,1200)),.4)
    assert instance.calls==5 and len(results)==1
    assert results[0]['confidence']==.81

def test_tiling_disabled_for_small_images():
    class Mock:
        calls=0
        def predict(self, image, conf):
            self.calls+=1
            return []
    m=Mock()
    detect_still_with_tiles(m,Image.new('RGB',(640,480)),.4)
    assert m.calls==1
