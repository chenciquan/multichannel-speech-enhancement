"""Analytic frequency/scale checks for the fixed DSP diagnostic, not method training."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mcse.spatial_noise import highpass40
spec=importlib.util.spec_from_file_location('noise_spectrum',ROOT/'scripts/analyze_noise_spectrum.py')
spectrum=importlib.util.module_from_spec(spec);spec.loader.exec_module(spectrum)


class SpectralControlTests(unittest.TestCase):
    def test_fixed_filter_removes16hz_preserves400hz_and_length(self):
        t=np.arange(48000)/16000
        lo=np.sin(2*np.pi*16*t);hi=np.sin(2*np.pi*400*t)
        y=highpass40(lo+hi)
        middle=slice(8000,-8000)
        lowgain=float(y[middle]@lo[middle]/(lo[middle]@lo[middle]))
        highgain=float(y[middle]@hi[middle]/(hi[middle]@hi[middle]))
        self.assertEqual(y.shape,t.shape)
        self.assertLess(abs(lowgain),.001)
        self.assertAlmostEqual(highgain,1,places=5)

    def test_filter_does_not_normalize_and_processes_channels_consistently(self):
        x=np.random.default_rng(67).normal(size=(12000,6))*.1
        y=highpass40(x)
        np.testing.assert_allclose(highpass40(x*.25),y*.25,atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(y[:,4],highpass40(x[:,4]),atol=1e-12)

    def test_parseval_band_fractions_have_known_answer(self):
        t=np.arange(16000)/16000
        result=spectrum.spectrum_summary(2*np.sin(2*np.pi*16*t)+np.sin(2*np.pi*1600*t))
        self.assertAlmostEqual(result['energy_below20hz_fraction'],.8,places=12)
        self.assertAlmostEqual(sum(result['band_energy_fractions'].values()),1,places=12)
        self.assertEqual(result['nonDC_peak_frequency_hz'],16)

    def test_filter_rejects_invalid_audio(self):
        for x in (np.ones(30),np.full(1000,np.inf),np.ones((1000,2,2))):
            with self.subTest(shape=x.shape),self.assertRaises(ValueError):highpass40(x)
