from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

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

        frompos = np.array([
            [1, 3],
            [2, 99],
            [-6, -5],
            [30, 21],
            [1, 0],
        ], dtype=float)
        topos = tranmap.tranForward(frompos)
        self.assertTrue(np.allclose(topos, frompos))
        rtpos = tranmap.tranInverse(topos)
        self.assertTrue(np.allclose(frompos, rtpos * zoomfac))

        with self.assertRaises(AssertionError):
            self.checkRoundTrip(tranmap, frompos)

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

        frompos = np.array([
            [1, 3],
            [2, 99],
            [-6, -5],
            [30, 21],
            [1, 0],
        ], dtype=float)
        topos = tranmap.tranForward(frompos)
        self.assertTrue(np.allclose(topos, frompos * zoomfac))

        self.checkRoundTrip(tranmap, frompos)


if __name__ == "__main__":
    unittest.main()
