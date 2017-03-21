from __future__ import absolute_import, division, print_function
from math import pi
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestNormMap(MappingTestCase):

    def test_NormMapFrame(self):
        """Check NormMap(Frame): output = input
        """
        normmap = astshim.NormMap(astshim.Frame(2))
        self.assertEqual(normmap.getClass(), "NormMap")
        self.assertEqual(normmap.getNin(), 2)
        self.assertEqual(normmap.getNout(), 2)

        self.checkBasicSimplify(normmap)
        self.checkCopy(normmap)
        self.checkPersistence(normmap)

        pin = np.array([
            [100.0, -100.0],
            [2000.0, -1000.0],
            [30000.0, -20000.0],
        ], dtype=float)
        pout = normmap.tranForward(pin)
        assert_allclose(pout, pin)

    def testNormMapMap(self):
        """Check that NormMap(Mapping) is an error"""
        with self.assertRaises(Exception):
            astshim.NormMap(astshim.UnitMap(1))

    def test_NormMapSkyFrame(self):
        """Check NormMap(SkYFrame):

        longitude wrapped to [0, 2 pi]
        if pi < |latitude| < 2 pi:
            offset longitude by pi and set latitutude = 2 pi - latitude
        else:
            wrap pi to range [-pi, pi]

        This test intentionally stays a small delta away from boundaries (except 0)
        because the expected behavior is not certain and not important
        """
        normmap = astshim.NormMap(astshim.SkyFrame())
        self.assertEqual(normmap.getClass(), "NormMap")
        self.assertEqual(normmap.getNin(), 2)
        self.assertEqual(normmap.getNout(), 2)

        self.checkBasicSimplify(normmap)
        self.checkPersistence(normmap)

        # I'm not sure why 100 is needed; I expected ~10 (2 pi)
        eps = 100 * sys.float_info.epsilon

        pin = np.array([
            [0, 0],
            [-eps, 0],  # lon out of range
            [2 * pi - eps, 0],
            [2 * pi + eps, 0],  # lon out of range
            [0, -pi / 2 + eps],
            [0, -pi / 2 - eps],  # lat too small; offset lat by pi
            [0, pi / 2 - eps],
            [0, pi / 2 + eps],  # lat too big; offset lat by pi
        ], dtype=float)
        despout = np.array([
            [0, 0],
            [2 * pi - eps, 0],
            [2 * pi - eps, 0],
            [eps, 0],
            [0, -pi / 2 + eps],
            [pi, -pi / 2 + eps],
            [0, pi / 2 - eps],
            [pi, pi / 2 - eps],
        ])
        pout = normmap.tranForward(pin)
        assert_allclose(pout, despout)


if __name__ == "__main__":
    unittest.main()
