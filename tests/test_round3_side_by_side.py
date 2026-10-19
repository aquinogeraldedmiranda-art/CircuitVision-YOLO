"""Round 3 installation never replaces the active Round 2 files."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from backend import model_archive as m

class Round3Separation(unittest.TestCase):
    def test_missing_keeps_round2(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'models';folder.mkdir()
            (folder/'best.onnx').write_bytes(b'approved round2')
            self.assertFalse(m.install_experimental_round3(Path(d)))
            self.assertEqual((folder/'best.onnx').read_bytes(),b'approved round2')

    def test_invalid_zip_refused_and_round2_safe(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'models';folder.mkdir()
            (folder/'best.onnx').write_bytes(b'approved round2')
            (folder/m.ROUND3_ARCHIVE_NAME).write_bytes(b'bad model')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                m.install_experimental_round3(Path(d))
            self.assertEqual((folder/'best.onnx').read_bytes(),b'approved round2')

    def test_valid_archive_is_installed_separately(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'models';folder.mkdir()
            (folder/'best.onnx').write_bytes(b'approved round2')
            old=['Resistor','LED','LDR','Power Module']+[f'Existing {i}' for i in range(4,57)]
            names=old+[f'New {i}' for i in range(24)]
            files={'best.pt':b'pt','best.onnx':b'experimental',
                   'classes.json':json.dumps({str(i):n for i,n in enumerate(names)}).encode(),
                   'TRAINING_REPORT.json':json.dumps({'old_classes':old,'all_classes':names}).encode(),
                   'DATASET_SOURCES.txt':b'credits'}
            archive=folder/m.ROUND3_ARCHIVE_NAME
            with zipfile.ZipFile(archive,'w') as z:
                for k,v in files.items():z.writestr(k,v)
            hashes={k:hashlib.sha256(files[k]).hexdigest() for k in m.ROUND3_FILES_SHA256}
            with patch.object(m,'ROUND3_ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()),patch.object(m,'ROUND3_FILES_SHA256',hashes):
                self.assertTrue(m.install_experimental_round3(Path(d)))
                self.assertEqual((folder/'best.onnx').read_bytes(),b'approved round2')
                self.assertEqual((folder/'round3.onnx').read_bytes(),b'experimental')
                self.assertTrue(m.install_experimental_round3(Path(d)))

if __name__=='__main__':unittest.main()
