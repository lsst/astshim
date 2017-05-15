from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestTranMap(MappingTestCase):

    def test_TranMapNotSymmetric(self):
        zoomfac = 0.5
        unitMap = astshim.UnitMap(2)
        zoomMap = astshim.ZoomMap(2, zoomfac)
        tranmap = astshim.TranMap(unitMap, zoomMap)
        # adding to a TranMap increases by 1
        self.assertEqual(unitMap.getRefCount(), 2)
        # adding to a TranMap increases by 1
        self.assertEqual(zoomMap.getRefCount(), 2)

        self.assertIsInstance(tranmap, astshim.TranMap)
        self.assertIsInstance(tranmap, astshim.Mapping)
        self.assertEqual(tranmap.getNin(), 2)
        self.assertEqual(tranmap.getNout(), 2)

        self.checkCopy(tranmap)
        self.checkPersistence(tranmap)

        indata = np.array([
            [1, 3],
            [2, 99],
            [-6, -5],
            [30, 21],
            [1, 0],
        ], dtype=float)
        outdata = tranmap.tranForward(indata)
        assert_allclose(outdata, indata)
        outdata_roundtrip = tranmap.tranInverse(outdata)
        assert_allclose(indata, outdata_roundtrip * zoomfac)

        with self.assertRaises(AssertionError):
            self.checkRoundTrip(tranmap, indata)

    def test_TranMapSymmetric(self):
        zoomfac = 0.53
        tranmap = astshim.TranMap(astshim.ZoomMap(
            2, zoomfac), astshim.ZoomMap(2, zoomfac))
        self.assertIsInstance(tranmap, astshim.TranMap)
        self.assertIsInstance(tranmap, astshim.Mapping)
        self.assertEqual(tranmap.getNin(), 2)
        self.assertEqual(tranmap.getNout(), 2)

        self.checkCopy(tranmap)
        self.checkPersistence(tranmap)

        indata = np.array([
            [1, 3],
            [2, 99],
            [-6, -5],
            [30, 21],
            [1, 0],
        ], dtype=float)
        outdata = tranmap.tranForward(indata)
        assert_allclose(outdata, indata * zoomfac)

        self.checkRoundTrip(tranmap, indata)


if __name__ == "__main__":
    unittest.main()
