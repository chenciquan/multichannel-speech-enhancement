import unittest
import numpy as np
from mcse.reverberation import full_speech,continuous_noise,schroeder_t20


class RoomConvolutionTests(unittest.TestCase):
    def test_last_spoken_sample_and_tail_are_retained(self):
        x=np.zeros(20);x[-1]=2;h=np.zeros((5,6));h[0]=1;h[-1]=.3
        y=full_speech(x,h)
        self.assertEqual(y.shape,(24,6));np.testing.assert_allclose(y[-1],.6,atol=1e-14)
        np.testing.assert_allclose(y[19],2,atol=1e-14)

    def test_noise_preroll_keeps_steady_background_through_tail(self):
        h=np.tile(np.array([1,.5,.25,.125])[:,None],(1,6))
        y=continuous_noise(np.ones(60),h,20,40)
        np.testing.assert_allclose(y,1.875,atol=1e-14)

    def test_noise_rejects_missing_history_or_length(self):
        h=np.ones((10,6))
        with self.assertRaises(ValueError):continuous_noise(np.ones(20),h,2,18)
        with self.assertRaises(ValueError):continuous_noise(np.ones(20),h,10,11)

    def test_exponential_decay_has_expected_t60(self):
        t=np.arange(16000)/16000;h=10**(-3*t/.3)
        self.assertAlmostEqual(schroeder_t20(h),.3,places=4)

    def test_invalid_audio_is_not_propagated(self):
        with self.assertRaises(ValueError):full_speech([np.nan],np.ones((2,6)))
        with self.assertRaises(ValueError):full_speech([1],np.ones((2,1)))
