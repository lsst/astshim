from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestUnitNormMap(MappingTestCase):

    def test_UnitNormMapBasics(self):
        """Test basics of UnitNormMap including applyForward
        """
        for nin in (1, 2, 3):
            center = np.array([-1, 1, 2][0:nin], dtype=float)
            unitnormmap = astshim.UnitNormMap(center)
            self.assertEqual(unitnormmap.className, "UnitNormMap")
            self.assertEqual(unitnormmap.nIn, nin)
            self.assertEqual(unitnormmap.nOut, nin + 1)
            self.assertFalse(unitnormmap.isLinear)

            self.checkBasicSimplify(unitnormmap)
            self.checkCopy(unitnormmap)
            self.checkPersistence(unitnormmap)

            indata = np.array([
                [1.0, 2.0, -6.0, 30.0, 1.0],
                [3.0, 99.0, -5.0, 21.0, 0.0],
                [-5.0, 3.0, -7.0, 37.0, 0.0],
                [7.0, -23.0, -3.0, 45.0, 0.0],
            ], dtype=float)[0:nin]
            self.checkRoundTrip(unitnormmap, indata)

            outdata = unitnormmap.applyForward(indata)
            norm = outdata[-1]

            relindata = (indata.T - center).T
            pred_norm = np.linalg.norm(relindata, axis=0)
            assert_allclose(norm, pred_norm)

            pred_relindata = outdata[0:nin] * norm
            assert_allclose(relindata, pred_relindata)

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
            [1.0, 2.0, -6.0, 30.0, 1.0],
            [3.0, 99.0, -5.0, 21.0, 0.0],
            [-5.0, 3.0, -7.0, 37.0, 0.0],
            [7.0, -23.0, -3.0, 45.0, 0.0],
        ], dtype=float)
        unm1 = astshim.UnitNormMap(center1)
        unm1inv = unm1.getInverse()
        unm2 = astshim.UnitNormMap(center2)
        unm2inv = unm2.getInverse()
        shiftmap = astshim.ShiftMap(shift)
        winmap_unitscale = astshim.WinMap(
            np.zeros(3), shift, np.ones(3), np.ones(3) + shift)
        winmap_notunitscale = astshim.WinMap(
            np.zeros(3), shift, np.ones(3), np.ones(3) * 2 + shift)

        for map1, map2, pred_simplified_class_name in (
            (unm1, unm2inv, "WinMap"),  # ShiftMap gets simplified to WinMap
            (shiftmap, unm1, "UnitNormMap"),
            (winmap_unitscale, unm1, "UnitNormMap"),
            (winmap_notunitscale, unm1, "SeriesMap"),
            (unm1inv, shiftmap, "UnitNormMap"),
            (unm1inv, winmap_unitscale, "UnitNormMap"),
            (unm1inv, winmap_notunitscale, "SeriesMap"),
        ):
            cmpmap = map1.then(map2)
            self.assertEqual(map1.nIn, cmpmap.nIn)
            self.assertEqual(map2.nOut, cmpmap.nOut)
            cmpmap_simp = cmpmap.simplify()
            self.assertEqual(cmpmap_simp.className, pred_simplified_class_name)
            self.assertEqual(cmpmap.nIn, cmpmap_simp.nIn)
            self.assertEqual(cmpmap.nOut, cmpmap_simp.nOut)
            testptview = np.array(testpoints[0:cmpmap.nIn])
            assert_allclose(cmpmap.applyForward(
                testptview), cmpmap_simp.applyForward(testptview))


if __name__ == "__main__":
    unittest.main()
