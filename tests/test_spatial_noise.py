"""Known-constraint tests for spatial diagnostics, independent of model results."""
import json
from pathlib import Path
import unittest
import numpy as np

import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mcse.spatial_noise import sphere_directions,conditioned_source_signals,noise_fields,spatial_coherence


class SpatialNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=json.loads((ROOT/"configs/g1_geometry.json").read_text())

    def test_sphere_directions_are_distinct_unit_vectors(self):
        u=sphere_directions()
        self.assertEqual(u.shape,(12,3))
        np.testing.assert_allclose(np.linalg.norm(u,axis=1),1,atol=1e-14)
        self.assertEqual(len(np.unique(u,axis=0)),12)

    def test_conditional_sources_sum_to_fixed_noise(self):
        n=np.random.default_rng(19).normal(size=8000)
        p=conditioned_source_signals(n,12,np.random.default_rng(41))
        np.testing.assert_allclose(p.sum(axis=0),n,atol=2e-14,rtol=1e-13)
        self.assertGreater(float(np.std(p[0]-p[1])),.1)

    def test_noise_fields_preserve_exact_reference_and_are_deterministic(self):
        n=np.random.default_rng(7).normal(size=4096)*.01
        a,detail=noise_fields(n,self.geometry['microphone_positions_m'],self.geometry['target_direction_unit'],73)
        b,_=noise_fields(n,self.geometry['microphone_positions_m'],self.geometry['target_direction_unit'],73)
        self.assertEqual(len(a),4)
        for name,field in a.items():
            self.assertEqual(field.shape,(4224,6))
            np.testing.assert_array_equal(field[:,4],np.pad(n,(64,64)))
            np.testing.assert_array_equal(field,b[name])
            self.assertLess(detail['reference_roundoff_errors_before_correction'][name],1e-10)
        self.assertGreater(np.linalg.norm(a['original_point'][:,0]-a['conditioned_12wave'][:,0]),.01)

    def test_measured_coherence_distinguishes_known_spatial_fields(self):
        n=np.random.default_rng(87).normal(size=32000)
        fields,_=noise_fields(n,self.geometry['microphone_positions_m'],self.geometry['target_direction_unit'],49)
        point=spatial_coherence(fields['original_point'])['mean']
        mixed=spatial_coherence(fields['conditioned_12wave'])['mean']
        self.assertGreater(point,.95)
        self.assertLess(mixed,point-.2)

    def test_invalid_sources_are_rejected(self):
        for n,count in ((np.ones(30),12),(np.ones((800,2)),12),(np.full(800,np.nan),12),(np.ones(800),1)):
            with self.subTest(count=count), self.assertRaises(ValueError):
                conditioned_source_signals(n,count,np.random.default_rng(3))
