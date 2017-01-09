from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestUnitNormMap(MappingTestCase):

    def test_UnitNormMapBasics(self):
        """Test basics of UnitNormMap including tran
        """
        for nin in (1, 2, 3):
            center = np.array([-1, 1, 2][0:nin], dtype=float)
            unitnormmap = astshim.UnitNormMap(center)
            self.assertEqual(unitnormmap.getClass(), "UnitNormMap")
            self.assertEqual(unitnormmap.getNin(), nin)
            self.assertEqual(unitnormmap.getNout(), nin+1)
            self.assertFalse(unitnormmap.getIsLinear())

            self.checkBasicSimplify(unitnormmap)
            self.checkCast(unitnormmap, goodType=astshim.Mapping, badType=astshim.ZoomMap)
            self.checkCopy(unitnormmap)
            self.checkPersistence(unitnormmap)

            frompos = np.array([
                [1, 3, -5, 7],
                [2, 99, 3, -23],
                [-6, -5, -7, -3],
                [30, 21, 37, 45],
                [1, 0, 0, 0],
            ], dtype=float)
            frompos = np.array(frompos[:, 0:nin])
            self.checkRoundTrip(unitnormmap, frompos)

            topos = unitnormmap.tran(frompos)
            norm = topos[:, -1]

            relfrompos = frompos - center
            prednorm = np.linalg.norm(relfrompos, axis=1)
            self.assertTrue(np.allclose(norm, prednorm))

            predrelfrompos = (topos[:, 0:nin].T*norm).T
            self.assertTrue(np.allclose(relfrompos, predrelfrompos))

        # UnitNormMap must have at least one input
        with self.assertRaises(Exception):
            astshim.UnitNormMap([])

    def test_UnitNormMapSimplify(self):
        """Test advanced simplification of UnitNormMap

        Basic simplification is tested elsewhere.

        ShiftMap              + UnitNormMap(forward)            = UnitNormMap(forward)
        UnitNormMap(inverted) + ShiftMap                        = UnitNormMap(inverted)
        UnitNormMap(forward)  + non-equal UnitNormMap(inverted) = ShiftMap
        """
        center1 = [2, -1, 0]
        center2 = [-1, 6, 4]
        shift = [3, 7, -9]
        # an array of points, each of 4 axes, the max we'll need
        testpoints = np.array([
            [1, 3, -5, 7],
            [2, 99, 3, -23],
            [-6, -5, -7, -3],
            [30, 21, 37, 45],
            [1, 0, 0, 0],
        ], dtype=float)
        unm1 = astshim.UnitNormMap(center1)
        unm1inv = unm1.getInverse()
        unm2 = astshim.UnitNormMap(center2)
        unm2inv = unm2.getInverse()
        shiftmap = astshim.ShiftMap(shift)
        winmap_unitscale = astshim.WinMap(
            np.zeros(3), shift, np.ones(3), np.ones(3) + shift)
        winmap_notunitscale = astshim.WinMap(
            np.zeros(3), shift, np.ones(3), np.ones(3)*2 + shift)

        for map1, map2, des_simplified_class_name in (
            (unm1, unm2inv, "WinMap"),  # ShiftMap gets simplified to WinMap
            (shiftmap, unm1, "UnitNormMap"),
            (winmap_unitscale, unm1, "UnitNormMap"),
            (winmap_notunitscale, unm1, "CmpMap"),
            (unm1inv, shiftmap, "UnitNormMap"),
            (unm1inv, winmap_unitscale, "UnitNormMap"),
            (unm1inv, winmap_notunitscale, "CmpMap"),
        ):
            cmpmap = map2.of(map1)
            self.assertEqual(map1.getNin(), cmpmap.getNin())
            self.assertEqual(map2.getNout(), cmpmap.getNout())
            cmpmap_simp = cmpmap.simplify()
            self.assertEqual(cmpmap_simp.getClass(), des_simplified_class_name)
            self.assertEqual(cmpmap.getNin(), cmpmap_simp.getNin())
            self.assertEqual(cmpmap.getNout(), cmpmap_simp.getNout())
            testptview = np.array(testpoints[:, 0:cmpmap.getNin()])
            self.assertTrue(np.allclose(cmpmap.tran(testptview), cmpmap_simp.tran(testptview)))


if __name__ == "__main__":
    unittest.main()
