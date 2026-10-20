"""Decode failures should distinguish huge image dimensions from bad codecs.

The 6250x6250 PNG emulates a small-on-disk lab-components source photo:
the pixel count matters even when the compressed file is tiny.
"""
import io
import struct
import zlib

import pytest
from fastapi import HTTPException
from PIL import Image

from backend.main import decode_image


def png_chunk(kind,payload):
    return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)


def oversized_png_header():
    ihdr=struct.pack('>IIBBBBB',6250,6250,8,2,0,0,0)
    return b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',ihdr)+png_chunk(b'IDAT',zlib.compress(b''))+png_chunk(b'IEND',b'')


def test_oversized_png_reports_resize_instead_of_invalid_format():
    with pytest.raises(HTTPException) as e:
        decode_image(oversized_png_header())
    assert e.value.status_code==413
    assert 'too large' in e.value.detail.lower()


def test_normal_png_is_accepted():
    raw=io.BytesIO()
    Image.new('RGBA',(70,40),(0,200,130,150)).save(raw,'PNG')
    image=decode_image(raw.getvalue())
    assert image.size==(70,40)
    assert image.mode=='RGB'


def test_unreadable_bytes_are_rejected():
    with pytest.raises(HTTPException) as e:
        decode_image(b'not a picture')
    assert e.value.status_code==415
    assert 'could not be read' in e.value.detail
