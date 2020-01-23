import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestTranMap(MappingTestCase):

    def test_TranMapNotSymmetric(self):
        zoomfac = 0.5
        unitMap = ast.UnitMap(2)
        zoomMap = ast.ZoomMap(2, zoomfac)
        tranmap = ast.TranMap(unitMap, zoomMap)
        # adding to a TranMap increases by 1
        self.assertEqual(unitMap.getRefCount(), 2)
        # adding to a TranMap increases by 1
        self.assertEqual(zoomMap.getRefCount(), 2)

        self.assertIsInstance(tranmap, ast.TranMap)
        self.assertIsInstance(tranmap, ast.Mapping)
        self.assertEqual(tranmap.nIn, 2)
        self.assertEqual(tranmap.nOut, 2)

        self.checkCopy(tranmap)

        indata = np.array([
            [1.0, 2.0, -6.0, 30.0, 1.0],
            [3.0, 99.0, -5.0, 21.0, 0.0],
        ], dtype=float)
        outdata = tranmap.applyForward(indata)
        assert_allclose(outdata, indata)
        outdata_roundtrip = tranmap.applyInverse(outdata)
        assert_allclose(indata, outdata_roundtrip * zoomfac)

        with self.assertRaises(AssertionError):
            self.checkRoundTrip(tranmap, indata)
        self.checkMappingPersistence(tranmap, indata)

    def test_TranMapSymmetric(self):
        zoomfac = 0.53
        tranmap = ast.TranMap(ast.ZoomMap(
            2, zoomfac), ast.ZoomMap(2, zoomfac))
        self.assertIsInstance(tranmap, ast.TranMap)
        self.assertIsInstance(tranmap, ast.Mapping)
        self.assertEqual(tranmap.nIn, 2)
        self.assertEqual(tranmap.nOut, 2)

        self.checkCopy(tranmap)

        indata = np.array([
            [1.0, 2.0, -6.0, 30.0, 1.0],
            [3.0, 99.0, -5.0, 21.0, 0.0],
        ], dtype=float)
        outdata = tranmap.applyForward(indata)
        assert_allclose(outdata, indata * zoomfac)

        self.checkRoundTrip(tranmap, indata)
        self.checkMappingPersistence(tranmap, indata)


if __name__ == "__main__":
    unittest.main()
