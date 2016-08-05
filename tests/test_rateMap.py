from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestRateMap(MappingTestCase):

    def test_RateMapBasics(self):
        zoomfac = 0.523
        ratemap = astshim.RateMap(astshim.ZoomMap(2, zoomfac), 2, 2)
        self.assertIsInstance(ratemap, astshim.RateMap)
        self.assertIsInstance(ratemap, astshim.Mapping)
        self.assertEqual(ratemap.getNout(), 1)

        self.checkBasicSimplify(ratemap)
        self.checkCopy(ratemap)
        self.checkPersistence(ratemap)

        indata = np.array([
            [1.1, 2.2],
            [-43.5, 1309.31],
        ])
        outdata = ratemap.tran(indata)
        self.assertTrue(np.allclose(outdata, zoomfac))

    def test_RateMap2(self):
        zoomfac = 23.323
        ratemap = astshim.RateMap(astshim.ZoomMap(2, zoomfac), 2, 1)

        indata = np.array([
            [1.1, 2.2],
            [-43.5, 1309.31],
        ])
        outdata = ratemap.tran(indata)
        self.assertTrue(np.allclose(outdata, 0))


if __name__ == "__main__":
    unittest.main()
