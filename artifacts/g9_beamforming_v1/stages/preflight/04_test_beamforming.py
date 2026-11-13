import unittest
import numpy as np
import torch
from mcse.beamforming import mask_mvdr,oracle_delay_and_sum,_stft,_wave
from mcse.simulation import render_plane_wave,geometric_delays


class BeamformingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_stft_roundtrip_exact_length(self):
        x=np.random.default_rng(42).normal(size=16013).astype(np.float32)*.03
        y=_wave(_stft(torch.tensor(x,dtype=torch.float64)),len(x))
        self.assertEqual(len(y),len(x));np.testing.assert_allclose(y,x,atol=1e-7,rtol=1e-6)

    def test_identical_channels_preserve_reference(self):
        x=np.random.default_rng(43).normal(size=16000).astype(np.float32)*.03
        y,_=mask_mvdr(np.tile(x[:,None],(1,6)),x)
        np.testing.assert_allclose(y,x,atol=1e-6,rtol=1e-4)

    def test_silence_degenerate_covariance_finite(self):
        y,info=mask_mvdr(np.zeros((16000,6)),np.zeros(16000))
        np.testing.assert_array_equal(y,0);self.assertTrue(np.isfinite(info['weight_max_magnitude']))

    def test_oracle_delay_sign_and_reference(self):
        # Independent renderer: reversing the steering sign fails this target-only SI check.
        pos=np.array([[-.06,0,0],[-.03,0,0],[0,0,0],[.03,0,0],[.06,0,0],[.09,0,0]])
        direction=np.array([.7,np.sqrt(1-.7**2),0]);d=geometric_delays(pos,direction)
        t=np.arange(32000)/16000;x=sum(.03*np.sin(2*np.pi*f*t) for f in (500,1200,2200))
        field=render_plane_wave(x,d);y=oracle_delay_and_sum(field,d);s=field[:,4]
        target=np.dot(y,s)/np.dot(s,s)*s;score=10*np.log10(np.dot(target,target)/np.dot(y-target,y-target))
        self.assertGreater(score,25.)

    def test_invalid_shapes_and_nonfinite_rejected(self):
        for x,e in [(np.zeros((16000,5)),np.zeros(16000)),(np.zeros((16000,6)),np.zeros(15999)),
                    (np.full((16000,6),np.nan),np.zeros(16000))]:
            with self.assertRaises(ValueError):mask_mvdr(x,e)


if __name__=='__main__':unittest.main()
