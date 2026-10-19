"""Round-2 ZIP is accepted only with original approved hashes and 57 classes."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend import model_archive as m


class AccuracyRound2ArchiveTests(unittest.TestCase):
    def test_no_round2_keeps_original_model(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            md = root / 'models'
            md.mkdir()
            (md / 'best.onnx').write_bytes(b'round1')
            self.assertFalse(m.install_accuracy_round2(root))
            self.assertEqual((md / 'best.onnx').read_bytes(), b'round1')

    def test_bad_round2_fails_without_overwriting_original(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            md = root / 'models'
            md.mkdir()
            (md / 'best.onnx').write_bytes(b'round1')
            (md / m.ROUND2_ARCHIVE_NAME).write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                m.install_accuracy_round2(root)
            self.assertEqual((md / 'best.onnx').read_bytes(), b'round1')

    def test_round2_verifies_bytes_before_replacing_original(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            md = root / 'models'
            md.mkdir()
            (md / 'best.onnx').write_bytes(b'round1')
            names = {'0':'Resistor','1':'LED','2':'LDR','3':'Power Module'}
            names.update({str(i):f'Class {i}' for i in range(4,57)})
            items = {'best.onnx':b'new-round2-model',
                     'classes.json':json.dumps(names).encode(),
                     'best.pt':b'checkpoint', 'DATASET_SOURCES.txt':b'credits'}
            archive = md / m.ROUND2_ARCHIVE_NAME
            with zipfile.ZipFile(archive,'w') as z:
                for k,v in items.items():
                    z.writestr(k,v)
            file_hashes = {k:hashlib.sha256(items[k]).hexdigest()
                           for k in ('best.onnx','classes.json')}
            with patch.object(m,'ROUND2_ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()), \
                 patch.object(m,'ROUND2_FILES_SHA256',file_hashes):
                self.assertTrue(m.install_accuracy_round2(root))
                self.assertEqual((md / 'best.onnx').read_bytes(),b'new-round2-model')
                self.assertTrue(m.install_accuracy_round2(root))
                self.assertFalse((md / 'best.pt').exists())

    def test_wrong_classes_keeps_original(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            md = root / 'models'
            md.mkdir()
            (md / 'best.onnx').write_bytes(b'round1')
            items={'best.onnx':b'round2','classes.json':b'{"0":"Resistor"}',
                   'best.pt':b'pt','DATASET_SOURCES.txt':b'credits'}
            archive=md/m.ROUND2_ARCHIVE_NAME
            with zipfile.ZipFile(archive,'w') as z:
                for k,v in items.items():z.writestr(k,v)
            file_hashes = {k:hashlib.sha256(items[k]).hexdigest()
                           for k in ('best.onnx','classes.json')}
            with patch.object(m,'ROUND2_ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()), \
                 patch.object(m,'ROUND2_FILES_SHA256',file_hashes):
                with self.assertRaisesRegex(RuntimeError,'model classes mismatch'):
                    m.install_accuracy_round2(root)
                self.assertEqual((md / 'best.onnx').read_bytes(),b'round1')


if __name__=='__main__':
    unittest.main()
