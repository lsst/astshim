import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestRateMap(MappingTestCase):

    def test_RateMapBasics(self):
        zoomfac = 0.523
        ratemap = ast.RateMap(ast.ZoomMap(2, zoomfac), 2, 2)
        self.assertIsInstance(ratemap, ast.RateMap)
        self.assertIsInstance(ratemap, ast.Mapping)
        self.assertEqual(ratemap.nOut, 1)

        self.checkBasicSimplify(ratemap)
        self.checkCopy(ratemap)

        indata = np.array([
            [1.1, -43.5, -5.54],
            [2.2, 1309.31, 35.2],
        ])
        outdata = ratemap.applyForward(indata)
        assert_allclose(outdata, zoomfac)

        self.checkMappingPersistence(ratemap, indata)

    def test_RateMap2(self):
        zoomfac = 23.323
        ratemap = ast.RateMap(ast.ZoomMap(2, zoomfac), 2, 1)

        indata = np.array([
            [1.1, -43.5, -5.54],
            [2.2, 1309.31, 35.2],
        ])
        outdata = ratemap.applyForward(indata)
        assert_allclose(outdata, 0)

        self.checkMappingPersistence(ratemap, indata)


if __name__ == "__main__":
    unittest.main()
