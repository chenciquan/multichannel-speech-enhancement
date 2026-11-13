import unittest
import numpy as np
from mcse.geometry import rotate_azimuth,azimuth_degrees,wrap_degrees
from mcse.spatial_noise import direction


class AzimuthGeometryTests(unittest.TestCase):
    def test_front_rotates_to_declared_right(self):
        np.testing.assert_allclose(rotate_azimuth([0,1,0],90),[1,0,0],atol=1e-14)

    def test_all_legacy_target_groups_keep_the_declared_point_offset(self):
        for target in (-40,20,70):
            for relative in (-80,25):
                u=rotate_azimuth(direction(20+relative,-15),target-20)
                self.assertAlmostEqual(wrap_degrees(azimuth_degrees(u)-target),relative,places=10)

    def test_target_and_all_noise_sources_undergo_same_rigid_rotation(self):
        rng=np.random.default_rng(829);sources=rng.normal(size=(16,3));sources/=np.linalg.norm(sources,axis=1)[:,None]
        for delta in (-130,-60,50,200):
            target=rotate_azimuth(direction(20,10),delta)
            np.testing.assert_allclose(target,direction(20+delta,10),atol=1e-14)
            moved=np.stack([rotate_azimuth(u,delta) for u in sources])
            np.testing.assert_allclose(moved@moved.T,sources@sources.T,atol=1e-14)
            np.testing.assert_allclose(moved@target,sources@direction(20,10),atol=1e-14)

    def test_composition_and_inverse_preserve_unit_direction(self):
        u=direction(173,-27)
        np.testing.assert_allclose(rotate_azimuth(rotate_azimuth(u,70),-70),u,atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(rotate_azimuth(u,70)),1.,places=14)

    def test_invalid_direction_is_rejected(self):
        for u in ([1,1,0],[np.nan,0,1],[1,0]):
            with self.assertRaises(ValueError):rotate_azimuth(u,20)
