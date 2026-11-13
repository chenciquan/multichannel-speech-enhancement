import io,unittest
import numpy as np
import soundfile as sf
import torch
from mcse.components import component_metrics,pcm24_field,decompose_fixed_mvdr
from mcse.beamforming import mask_mvdr


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        rng=np.random.default_rng(421);return rng.normal(size=16000)*.02,rng.normal(size=16000)*.02

    def test_known_gain_and_noise_energy_accounting(self):
        s,n=self.fixture();r=component_metrics(s,n,.5*s,.1*n)
        self.assertAlmostEqual(r['speech_error_nmse'],.25);self.assertAlmostEqual(r['target_gain_db'],-6.020599913)
        self.assertAlmostEqual(r['noise_reduction_db'],20.);self.assertLess(r['snr_accounting_error_db'],1e-10)
        self.assertGreater(r['target_component_si_sdr_db'],100.)

    def test_identity_components(self):
        s,n=self.fixture();r=component_metrics(s,n,s,n)
        self.assertEqual(r['speech_error_nmse'],0.);self.assertEqual(r['target_gain_db'],0.);self.assertEqual(r['noise_reduction_db'],0.)

    def test_perfect_rejection_floor_flagged(self):
        s,n=self.fixture();r=component_metrics(s,n,s,np.zeros_like(n))
        self.assertTrue(r['output_noise_power_floored']);self.assertGreater(r['noise_reduction_db'],200.)

    def test_speech_cancellation_not_silently_omitted(self):
        s,n=self.fixture();r=component_metrics(s,n,np.zeros_like(s),n)
        self.assertEqual(r['speech_error_nmse'],1.);self.assertTrue(r['output_speech_power_floored']);self.assertEqual(r['target_component_si_sdr_status'],'undefined')

    def test_invalid_and_zero_reference_rejected(self):
        s,n=self.fixture()
        for values in [(s,n,s[:-1],n),(s,n,s,np.full_like(n,np.nan)),(s,np.zeros_like(n),s,n),(s.astype(complex),n,s,n)]:
            with self.assertRaises(ValueError):component_metrics(*values)

    def test_pcm24_channel_matches_independent_mono_roundtrip(self):
        s,n=self.fixture();a=np.stack([s+i*.001 for i in range(6)],axis=1);q=pcm24_field(a)
        b=io.BytesIO();sf.write(b,a[:,4],16000,format='WAV',subtype='PCM_24');b.seek(0)
        np.testing.assert_array_equal(q[:,4],sf.read(b,dtype='float32')[0])

    def test_exact_filter_reproduction_and_linearity(self):
        s,n=self.fixture();rng=np.random.default_rng(422);clean=pcm24_field(np.tile(s[:,None],(1,6)));x=clean+rng.normal(size=clean.shape)*.02
        estimate=clean[:,4];expected,_=mask_mvdr(x,estimate);y,ys,yn,r=decompose_fixed_mvdr(x,estimate,clean)
        np.testing.assert_array_equal(y,expected);self.assertLess(r['linearity_max_abs'],1e-6)


if __name__=='__main__':unittest.main()
