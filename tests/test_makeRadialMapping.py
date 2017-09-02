
from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
import numpy.testing as npt

import astshim as ast
from astshim.test import MappingTestCase, makeForwardPolyMap


class TestMakeRadialMapping(MappingTestCase):
    def setUp(self):
        self.in_data_full = np.array([
            [0.0, 0.1, 1.1, 50.1],
            [1.45, -47.3, 0.546, 37.3],
            [0.34, 54.3, 16.2, -55.5],
        ], dtype=float)

    def test_MakeRadialMappingForward(self):
        """Test a radial_map mapping that only has a forward transform
        """
        coeff_f = np.array([
            [5.0, 1, 1],
            [-0.12, 1, 2],
        ])
        mapping1d = ast.PolyMap(coeff_f, 1)
        self.assertTrue(mapping1d.hasForward)
        self.assertFalse(mapping1d.hasInverse)

        # if center = [0.0] then the radial_map mapping is identical to mapping1d
        radial_map = ast.makeRadialMapping([0], mapping1d)
        self.assertTrue(radial_map.hasForward)
        self.assertFalse(radial_map.hasInverse)
        in_data1 = self.in_data_full[0]
        npt.assert_allclose(radial_map.applyForward(in_data1), mapping1d.applyForward(in_data1))

        for center in (
            [0.0],
            [1.1],
            [0.0, 0.0],
            [-5.5, 4.7],
            [0.0, 0.0, 0.0],
            [1.1, 2.2, -3.3],
        ):
            naxes = len(center)
            center_reshaped = np.expand_dims(center, 1)
            in_data = self.in_data_full[0:naxes]

            radial_map = ast.makeRadialMapping(center, mapping1d)
            self.assertTrue(radial_map.hasForward)
            self.assertFalse(radial_map.hasInverse)

            # compute desired output
            in_from_center = in_data - center_reshaped
            in_norm = np.linalg.norm(in_from_center, axis=0)
            unit_vector = np.where(in_norm != 0, in_from_center / in_norm, in_from_center)
            out_norm = mapping1d.applyForward(in_norm)
            out_from_center = unit_vector * out_norm
            desired_out_data = out_from_center + center_reshaped

            out_data = radial_map.applyForward(in_data)
            npt.assert_allclose(out_data, desired_out_data)

    def test_MakeRadialMappingInvertible(self):
        """Test makeRadialMapping on a mapping that has an accurate inverse"""
        zoom = 5.5
        mapping1d = ast.ZoomMap(1, zoom)
        self.assertTrue(mapping1d.hasForward)
        self.assertTrue(mapping1d.hasInverse)

        for center in (
            [0.0],
            [1.1],
            [0.0, 0.0],
            [-5.5, 4.7],
            [0.0, 0.0, 0.0],
            [1.1, 2.2, -3.3],
        ):
            naxes = len(center)
            center_reshaped = np.expand_dims(center, 1)
            in_data = self.in_data_full[0:naxes]

            radial_map = ast.makeRadialMapping(center, mapping1d)
            self.assertTrue(radial_map.hasForward)
            self.assertTrue(radial_map.hasInverse)
            self.checkRoundTrip(radial_map, in_data)

            # compute desired output
            in_from_center = in_data - center_reshaped
            in_norm = np.linalg.norm(in_from_center, axis=0)
            unit_vector = np.where(in_norm != 0, in_from_center / in_norm, in_from_center)
            out_norm = in_norm * zoom
            out_from_center = unit_vector * out_norm
            desired_out_data = out_from_center + center_reshaped

            out_data = radial_map.applyForward(in_data)
            npt.assert_allclose(out_data, desired_out_data)

    def test_MakeRadialMappingErrorHandling(self):
        """Test error handling in makeRadialMapping"""
        for bad_nin in (1, 2, 3):
            for bad_nout in (1, 2, 3):
                if bad_nin == bad_nout == 1:
                    continue  # the only valid case
                bad_mapping1d = makeForwardPolyMap(bad_nin, bad_nout)
                with self.assertRaises(ValueError):
                    ast.makeRadialMapping([0.0], bad_mapping1d)

        mapping1d = ast.ZoomMap(1, 5.5)
        with self.assertRaises(RuntimeError):
            ast.makeRadialMapping([], mapping1d)


if __name__ == "__main__":
    unittest.main()
