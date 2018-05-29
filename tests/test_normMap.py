from __future__ import absolute_import, division, print_function
from math import pi
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestNormMap(MappingTestCase):

    def test_NormMapFrame(self):
        """Check NormMap(Frame): output = input
        """
        normmap = ast.NormMap(ast.Frame(2))
        self.assertEqual(normmap.className, "NormMap")
        self.assertEqual(normmap.nIn, 2)
        self.assertEqual(normmap.nOut, 2)

        self.checkBasicSimplify(normmap)
        self.checkCopy(normmap)

        indata = np.array([
            [100.0, 2000.0, 3000.0],
            [-100.0, -1000.0, -2000.0],
        ], dtype=float)
        outdata = normmap.applyForward(indata)
        assert_allclose(outdata, indata, atol=1e-12)

        self.checkMappingPersistence(normmap, indata)

    def testNormMapMap(self):
        """Check that NormMap(Mapping) is an error"""
        with self.assertRaises(TypeError):
            ast.NormMap(ast.UnitMap(1))

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
        normmap = ast.NormMap(ast.SkyFrame())
        self.assertEqual(normmap.className, "NormMap")
        self.assertEqual(normmap.nIn, 2)
        self.assertEqual(normmap.nOut, 2)

        self.checkBasicSimplify(normmap)

        # I'm not sure why 100 is needed; I expected ~10 (2 pi)
        eps = 100 * sys.float_info.epsilon

        indata = (np.array([
            [0, 0],
            [-eps, 0],  # lon out of range
            [2 * pi - eps, 0],
            [2 * pi + eps, 0],  # lon out of range
            [0, -pi / 2 + eps],
            [0, -pi / 2 - eps],  # lat too small; offset lat by pi
            [0, pi / 2 - eps],
            [0, pi / 2 + eps],  # lat too big; offset lat by pi
        ], dtype=float)).T.copy()  # applyForward can't accept a view
        pred_outdata = np.array([
            [0, 0],
            [2 * pi - eps, 0],
            [2 * pi - eps, 0],
            [eps, 0],
            [0, -pi / 2 + eps],
            [pi, -pi / 2 + eps],
            [0, pi / 2 - eps],
            [pi, pi / 2 - eps],
        ]).T
        outdata = normmap.applyForward(indata)
        assert_allclose(outdata, pred_outdata, atol=1e-12)

        self.checkMappingPersistence(normmap, indata)


if __name__ == "__main__":
    unittest.main()
