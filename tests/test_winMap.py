from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestWinMap(MappingTestCase):

    def test_WinMap(self):
        # a map describing a shift of [1.0, -0.5] followed by a zoom of 2
        winmap = ast.WinMap([0, 0], [1, 1], [1, -0.5], [3, 1.5])
        pred_shift = [1.0, -0.5]
        pred_zoom = 2.0
        self.assertIsInstance(winmap, ast.WinMap)
        self.assertEqual(winmap.nIn, 2)
        self.assertEqual(winmap.nOut, 2)

        self.checkBasicSimplify(winmap)
        self.checkCopy(winmap)

        indata = np.array([
            [0.0, 0.5, 1.0],
            [-3.0, 1.5, 0.13],
        ], dtype=float)
        pred_outdata = (indata.T * pred_zoom + pred_shift).T
        outdata = winmap.applyForward(indata)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(winmap, indata)
        self.checkMappingPersistence(winmap, indata)


if __name__ == "__main__":
    unittest.main()
