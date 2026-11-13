import unittest
import numpy as np
from mcse.asr_level import standardize_level


class LevelControlTests(unittest.TestCase):
    def test_rms_target_and_exact_binary_invariance(self):
        t=np.arange(1600,dtype=np.float64)/16000
        x=(.2*np.sin(2*np.pi*337*t)+.07*np.cos(2*np.pi*521*t)).astype('float32')
        normalized,info=standardize_level(x)
        self.assertFalse(info['peak_limited'])
        self.assertAlmostEqual(info['output_rms'],.05,places=8)
        for gain in (.25,.5,2.,4.):
            other,_=standardize_level(x*gain)
            self.assertEqual(other.tobytes(),normalized.tobytes())

    def test_peak_limit_and_binary_invariance(self):
        x=np.zeros(512,dtype='float32');x[50]=1.
        y,info=standardize_level(x)
        self.assertTrue(info['peak_limited']);self.assertLess(info['output_rms'],.05)
        self.assertAlmostEqual(float(y.max()),.95,places=7)
        self.assertEqual(standardize_level(x*.25)[0].tobytes(),y.tobytes())

    def test_silence_stays_silence(self):
        y,info=standardize_level(np.zeros(512,dtype='float32'))
        self.assertTrue(info['silent']);self.assertEqual(np.count_nonzero(y),0)

    def test_invalid_inputs(self):
        for value in (np.zeros(5,dtype='float64'),np.zeros((2,2),dtype='float32'),np.array([np.nan],dtype='float32'),np.array([],dtype='float32')):
            with self.assertRaises(ValueError):standardize_level(value)
        for target,peak in ((0,.95),(.9,.1),(.05,2.),(.05,float('nan'))):
            with self.assertRaises(ValueError):standardize_level(np.zeros(5,dtype='float32'),target,peak)


if __name__=='__main__':unittest.main()
