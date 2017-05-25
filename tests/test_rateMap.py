from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestRateMap(MappingTestCase):

    def test_RateMapBasics(self):
        zoomfac = 0.523
        ratemap = astshim.RateMap(astshim.ZoomMap(2, zoomfac), 2, 2)
        self.assertIsInstance(ratemap, astshim.RateMap)
        self.assertIsInstance(ratemap, astshim.Mapping)
        self.assertEqual(ratemap.nOut, 1)

        self.checkBasicSimplify(ratemap)
        self.checkCopy(ratemap)
        self.checkPersistence(ratemap)

        indata = np.array([
            [1.1, -43.5, -5.54],
            [2.2, 1309.31, 35.2],
        ])
        outdata = ratemap.tranForward(indata)
        assert_allclose(outdata, zoomfac)

    def test_RateMap2(self):
        zoomfac = 23.323
        ratemap = astshim.RateMap(astshim.ZoomMap(2, zoomfac), 2, 1)

        indata = np.array([
            [1.1, -43.5, -5.54],
            [2.2, 1309.31, 35.2],
        ])
        outdata = ratemap.tranForward(indata)
        assert_allclose(outdata, 0)


if __name__ == "__main__":
    unittest.main()
