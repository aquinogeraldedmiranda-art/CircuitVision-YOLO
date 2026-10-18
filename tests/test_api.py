"""Automated tests use a fake inference function. They do NOT assert model accuracy."""
from __future__ import annotations
import io
from fastapi.testclient import TestClient
from PIL import Image
from backend import main

client = TestClient(main.app)

def image(fmt='PNG', wh=(180,130)):
    b=io.BytesIO();Image.new('RGB',wh,'#335566').save(b,format=fmt);return b.getvalue()

def test_same_host_serves_website_and_api():
    assert client.get('/').status_code==200
    assert 'CircuitVision' in client.get('/').text
    for path in ('/app.js','/api.js','/styles.css','/config.js','/assets/favicon.svg'):
        assert client.get(path).status_code == 200
    for path in ('/backend/main.py','/models/best.onnx','/render.yaml'):
        assert client.get(path).status_code == 404

def test_health_missing_weights(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'CUSTOM_MODEL',str(tmp_path/'missing.onnx'))
    r=client.get('/api/health').json()
    assert r['ok'] is True and r['custom_model_available'] is False

def test_rejects_missing_and_bad_files():
    assert client.post('/api/detect',files={'image':('empty.jpg',b'','image/jpeg')}).status_code==400
    assert client.post('/api/detect',files={'image':('bad.jpg',b'not-jpeg','image/jpeg')}).status_code==415
    assert client.post('/api/detect',data={'confidence':'1.5'},files={'image':('ok.png',image(),'image/png')}).status_code==422
    assert client.post('/api/detect',data={'model':'../../../bad'},files={'image':('ok.png',image(),'image/png')}).status_code==422
    assert client.post('/api/detect',data={'model':'yolo11n'},files={'image':('ok.png',image(),'image/png')}).status_code==422

def test_inference_contract_from_mock(monkeypatch):
    calls=[]
    def fake(img,model,conf):
        calls.append((img.size,model,conf))
        return [{'label':'Power Module','class_id':0,'confidence':.64,'bbox':[.1,.2,.7,.9]}]
    monkeypatch.setattr(main,'infer',fake)
    r=client.post('/api/detect',data={'model':'custom','confidence':'.55'},files={'image':('x.jpg',image('JPEG'),'image/jpeg')})
    assert r.status_code==200
    data=r.json()
    assert data['image_width']==180 and data['image_height']==130
    assert data['detections'][0]['label']=='Power Module'
    assert calls==[((180,130),'custom',.55)]

def test_untrained_weights_return_503(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'CUSTOM_MODEL',str(tmp_path/'none.pt'))
    r=client.post('/api/detect',files={'image':('valid.png',image(),'image/png')})
    assert r.status_code == 503
    assert 'weights' in r.json()['detail']
