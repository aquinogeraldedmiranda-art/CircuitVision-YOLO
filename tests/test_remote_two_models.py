"""Smoke-test both user-selected CircuitVision YOLOv11 detectors on the live site.

This proves that the independently selected API routes load real ONNX weights.
It deliberately does not attempt to measure detection accuracy with fake images.
"""
import json, os, struct, urllib.request, urllib.error, zlib

base=os.environ.get('CIRCUITVISION_BASE','https://circuitvision-yolo-lab.onrender.com')
with urllib.request.urlopen(base+'/api/health',timeout=150) as reply:
    h=json.load(reply)
print('Health model choice:',json.dumps({
    k:h.get(k) for k in ['deployment_commit','yolov11_1_available',
                         'yolov11_2_available','yolov11_2_class_count']}),
    flush=True)
assert h.get('yolov11_1_available') is True, 'YOLOv11 (1) is not available'
assert h.get('yolov11_2_available') is True, 'YOLOv11 (2) is not installed'
assert h.get('yolov11_2_class_count')==87, 'Wrong candidate class count'

w,hgt=96,96
def chunk(k,payload):
    return struct.pack('!I',len(payload))+k+payload+struct.pack('!I',zlib.crc32(k+payload)&0xffffffff)
image=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',w,hgt,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\x00'+b'\x80\x80\x80'*w)*hgt))+chunk(b'IEND',b'')
boundary=b'circuitvision-yolov11-choice-test'
def request_model(choice):
    def field(name, value):
        return (b'--'+boundary+b'\r\nContent-Disposition: form-data; name="'+name+b'"\r\n\r\n'+value+b'\r\n')
    form=field(b'model',choice.encode())+field(b'live',b'true')+field(b'confidence',b'0.35')
    form+=(b'--'+boundary+b'\r\nContent-Disposition: form-data; name="image"; filename="control.png"\r\nContent-Type: image/png\r\n\r\n'+image+b'\r\n--'+boundary+b'--\r\n')
    req=urllib.request.Request(base+'/api/detect',data=form,
        headers={'Content-Type':'multipart/form-data; boundary='+boundary.decode()},
        method='POST')
    try:
        with urllib.request.urlopen(req,timeout=160) as response:
            body=json.load(response)
    except urllib.error.HTTPError as exc:
        print('HTTP error',choice,exc.code,exc.read().decode(errors='replace')[:700],flush=True)
        raise
    assert isinstance(body.get('detections'),list), f'{choice}: invalid output'
    assert body.get('model')==('YOLOv11 (2)' if choice=='unified' else 'YOLOv11 (1)'),body
    print(choice,'OK; processing_ms=',body.get('processing_ms'),flush=True)

for name in ['auto','unified']:
    request_model(name)
print('Both production YOLOv11 models accepted independent inference requests.',flush=True)
