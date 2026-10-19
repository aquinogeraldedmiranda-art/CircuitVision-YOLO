"""Only an exact, intact student model ZIP may be installed."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from backend import model_archive as m


class StudentModelArchiveTests(unittest.TestCase):
    def test_no_archive_preserves_existing_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(m.install_uploaded_checkpoint(Path(tmp)))
            self.assertFalse((Path(tmp) / 'models' / 'best.onnx').exists())

    def test_bad_archive_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / 'models'
            model_dir.mkdir()
            (model_dir / m.ARCHIVE_NAME).write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                m.install_uploaded_checkpoint(root)
            self.assertFalse((model_dir / 'best.onnx').exists())

    def test_valid_pinned_archive_extracts_model_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / 'models'
            model_dir.mkdir()
            labels = {'0': 'Resistor', '1': 'LED', '2': 'LDR', '3': 'Power Module'}
            labels.update({str(i): f'Other class {i}' for i in range(4, 57)})
            fake_members = {
                'best.onnx': b'fake ONNX contents',
                'classes.json': json.dumps(labels).encode(),
                'best.pt': b'not needed for Render',
                'DATASET_SOURCES.txt': b'citation',
            }
            archive = model_dir / m.ARCHIVE_NAME
            with zipfile.ZipFile(archive, 'w') as z:
                for name, contents in fake_members.items():
                    z.writestr(name, contents)
            hashes = {name: hashlib.sha256(fake_members[name]).hexdigest()
                      for name in ('best.onnx', 'classes.json')}
            with patch.object(m, 'ARCHIVE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest()), \
                 patch.object(m, 'FILES_SHA256', hashes):
                self.assertTrue(m.install_uploaded_checkpoint(root))
                self.assertEqual((model_dir / 'best.onnx').read_bytes(), b'fake ONNX contents')
                self.assertEqual(json.loads((model_dir / 'classes.json').read_text()), labels)
                self.assertFalse((model_dir / 'best.pt').exists())
                self.assertTrue(m.install_uploaded_checkpoint(root))

    def test_bad_classes_fail_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / 'models'
            model_dir.mkdir()
            bad_labels = {'0': 'Resistor'}
            members = {'best.onnx': b'weights', 'classes.json': json.dumps(bad_labels).encode(),
                       'best.pt': b'pt', 'DATASET_SOURCES.txt': b'credits'}
            archive = model_dir / m.ARCHIVE_NAME
            with zipfile.ZipFile(archive, 'w') as z:
                for name, data in members.items(): z.writestr(name, data)
            hashes = {name: hashlib.sha256(members[name]).hexdigest()
                      for name in ('best.onnx', 'classes.json')}
            with patch.object(m, 'ARCHIVE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest()), \
                 patch.object(m, 'FILES_SHA256', hashes):
                with self.assertRaisesRegex(RuntimeError, 'class mapping'):
                    m.install_uploaded_checkpoint(root)
            self.assertFalse((model_dir / 'best.onnx').exists())


if __name__ == '__main__':
    unittest.main()
