"""Test YOLO11 ONNX geometry, confidence, NMS, and labels without requiring a model file."""
import numpy as np
from PIL import Image
from backend.onnx_inference import make_letterbox, decode_yolo11, parse_class_map, non_max_suppression

def test_names_parsing():
    assert parse_class_map('{0: "Power Module", 1: "Resistor"}') == {0:'Power Module',1:'Resistor'}
    assert parse_class_map('["A", "B"]') == {0:'A',1:'B'}
    assert parse_class_map('definitely invalid') == {}

def test_letterbox_and_boxes():
    im=Image.new('RGB',(200,100),(50,100,150))
    x,offset=make_letterbox(im,416,416)
    assert x.shape == (1,3,416,416)
    # Factor 2.08, resized image 416 x 208, centered with top offset 104.
    assert offset == (0,104,2.08,2.08)
    # Two duplicate power module boxes + non-overlapping resistor box.
    raw=np.zeros((1,6,8),dtype='float32')
    raw[:,0,:3] = [208,209,80] # cx
    raw[:,1,:3] = [208,209,208] # cy
    raw[:,2,:3] = [208,208,40] # bw
    raw[:,3,:3] = [104,104,40] # bh
    raw[:,4,:3] = [.85,.8,.15]
    raw[:,5,:3] = [.05,.1,.95]
    rows=decode_yolo11(raw,offset,200,100,{0:'Power Module',1:'Resistor'},.50)
    assert len(rows)==2
    assert rows[0]['label']=='Resistor' and rows[1]['label']=='Power Module'
    assert rows[1]['bbox']==[.25,.25,.75,.75]

def test_iou_class_specific_nms():
    b=np.array([[0,0,10,10],[1,1,10,10],[0,0,10,10]],np.float32)
    keep=non_max_suppression(b,np.array([.9,.8,.7]),np.array([0,0,1]))
    assert list(keep)==[0,2]

def test_yolo26_end_to_end_boxes():
    from backend.onnx_inference import decode_yolo26
    data=np.array([[[104,130,312,338,.86,2], [105,132,312,338,.07,2],
                    [20,20,20,30,.95,1], [200,200,300,300,.85,1]]],dtype=np.float32)
    result=decode_yolo26(data,(0,104,2.08,2.08),200,100,
                         {1:'Capacitor',2:'Power Module'},0.50)
    assert len(result)==2
    assert result[0]['label']=='Power Module'
    assert result[0]['bbox']==[.25,.125,.75,1.0]
    assert result[1]['label']=='Capacitor'

def test_pinned_pretrained_checksum_rejects_other_bytes():
    from backend.pretrained import verify_model
    assert not verify_model(b'fake weights')
