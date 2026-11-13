import argparse,hashlib,json,runpy,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mcse.runtime import validate_array,read_input,archived_json,load_system

class RuntimeContractTests(unittest.TestCase):
    def test_valid_real_audio_preserves_values(self):
        a=np.linspace(-.5,.5,512*6).reshape(512,6);b=validate_array(a,16000)
        self.assertEqual(b.dtype,np.float32);self.assertTrue(b.flags.c_contiguous);np.testing.assert_array_equal(b,a.astype('float32'))
        self.assertEqual(validate_array(np.zeros((64000,6)),16000).shape,(64000,6))
    def test_rate_channels_and_lengths_rejected(self):
        for a,sr in [(np.zeros((16000,1)),16000),(np.zeros((16000,5)),16000),(np.zeros((16000,7)),16000),(np.zeros((16000,6)),8000),(np.zeros((511,6)),16000),(np.zeros((64001,6)),16000)]:
            with self.subTest(shape=a.shape,sr=sr),self.assertRaises(ValueError):validate_array(a,sr)
    def test_nonfinite_overflow_and_complex_rejected(self):
        for value in (np.nan,np.inf,-np.inf,1e300):
            a=np.zeros((512,6));a[0,0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_array(a,16000)
        with self.assertRaises(ValueError):validate_array(np.zeros((512,6),dtype='complex64'),16000)
    def test_file_header_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'fixture.wav';sf.write(p,np.zeros((16000,2)),16000)
            with self.assertRaises(ValueError):read_input(p)
            sf.write(p,np.zeros((16000,6)),8000)
            with self.assertRaises(ValueError):read_input(p)
            sf.write(p,np.zeros((512,6)),16000);self.assertEqual(read_input(p).shape,(512,6))
    def test_registry_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'registry.json';archive=root/'archive.json';source.write_text('{"systems": []}')
            archive.write_text(json.dumps({'files':[{'source':'registry.json','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}))
            with patch('mcse.runtime.ROOT',root):
                self.assertEqual(archived_json(source,archive),{'systems':[]});source.write_text('{"systems": ["changed"]}')
                with self.assertRaises(ValueError):archived_json(source,archive)
    def test_adapter_hash_mismatch_never_loads_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'adapter.pt').write_bytes(b'tampered checkpoint bytes')
            registry={'test':{'kind':'adapter','path':'adapter.pt','checkpoint_sha256':'0'*64}}
            fake={'OFFLINE_CHECKPOINT':'base.pt','verify_upstream':lambda *args:{'checkpoint':{'sha256':'known'}},'load_model':lambda *args:self.fail('Unverified checkpoint must never load')}
            with patch('mcse.runtime.ROOT',root),patch('mcse.runtime.registered_systems',return_value=registry),patch('mcse.runtime.bridge',return_value=fake):
                with self.assertRaisesRegex(ValueError,'SHA256'):load_system('test','cpu')

class CLIRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cli=runpy.run_path(str(ROOT/'scripts/enhance_registered.py'),run_name='registered_cli_test')
    def test_existing_or_alias_outputs_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'input.wav';sf.write(p,np.zeros((512,6)),16000)
            for which in ('output','report','alias'):
                output=root/('out_'+which+'.wav');report=root/('report_'+which+'.json')
                if which=='output':output.write_bytes(b'preserve')
                elif which=='report':report.write_text('preserve')
                else:report=output
                args=argparse.Namespace(input=p,output=output,report=report,system='mcnet',device='cuda',list_systems=False)
                # Patch the function globals rather than an imported duplicate module.
                with patch.dict(self.cli['run'].__globals__,{'load_system':lambda *args: self.fail('Model must not load')}):
                    with self.assertRaises((FileExistsError,ValueError)):self.cli['run'](args)
                if which=='output':self.assertEqual(output.read_bytes(),b'preserve')
                if which=='report':self.assertEqual(report.read_text(),'preserve')
    def test_wrong_audio_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'mono.wav';sf.write(p,np.zeros(512),16000)
            args=argparse.Namespace(input=p,output=root/'out.wav',report=root/'out.json',system='mcnet',device='cuda',list_systems=False)
            with patch.dict(self.cli['run'].__globals__,{'load_system':lambda *args:self.fail('Model must not load')}):
                with self.assertRaises(ValueError):self.cli['run'](args)
            self.assertFalse(args.output.exists());self.assertFalse(args.report.exists())
    def test_unknown_system_does_not_load_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'six.wav';sf.write(p,np.zeros((512,6)),16000)
            args=argparse.Namespace(input=p,output=root/'out.wav',report=root/'out.json',system='unknown',device='cuda',list_systems=False)
            with patch.dict(self.cli['run'].__globals__,{'registered_systems':lambda:{'mcnet':{}},'load_system':lambda *args:self.fail('Unknown system must not load')}):
                with self.assertRaisesRegex(ValueError,'Unknown'):self.cli['run'](args)
            self.assertFalse(args.output.exists());self.assertFalse(args.report.exists())
