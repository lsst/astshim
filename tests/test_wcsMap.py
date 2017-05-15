from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestWcsMap(MappingTestCase):

    def test_WcsMap(self):
        # Test the Aitoff projection because it is locally flat at 0,0
        # unlike the tangent-plane projection which maps (0, pi/2) -> 0,0
        # and is not very well behaved at nearby points (as expected at a
        # pole).
        wcsmap = astshim.WcsMap(2, astshim.WcsType.AIT, 1, 2)
        self.assertIsInstance(wcsmap, astshim.WcsMap)
        self.assertIsInstance(wcsmap, astshim.Mapping)
        self.assertEqual(wcsmap.getNin(), 2)
        self.assertEqual(wcsmap.getNout(), 2)

        self.assertEqual(wcsmap.getNatLon(), 0)
        self.assertEqual(wcsmap.getNatLat(), 0)
        self.assertEqual(wcsmap.getPVMax(1), 4)
        self.assertEqual(wcsmap.getPVi_m(1, 0), 0)
        self.assertEqual(wcsmap.getPVi_m(1, 1), 0)
        self.assertEqual(wcsmap.getPVi_m(1, 2), 0)
        self.assertTrue(math.isinf(wcsmap.getPVi_m(1, 3)))
        self.assertTrue(math.isinf(wcsmap.getPVi_m(1, 4)))
        self.assertEqual(wcsmap.getPVMax(2), 0)
        self.assertEqual(wcsmap.getWcsAxis(1), 1)
        self.assertEqual(wcsmap.getWcsAxis(2), 2)
        self.assertEqual(wcsmap.getWcsType(), astshim.WcsType.AIT)

        self.checkBasicSimplify(wcsmap)
        self.checkCopy(wcsmap)
        self.checkPersistence(wcsmap)

        indata = np.array([
            [0, 0],
            [0.001, 0],
            [0, 0.001],
            [0, 1],
        ], dtype=float)
        pred_outdata = np.array([
            [0, 0],
            [0.001, 0],
            [0, 0.001],
            [0, 0.95885108],  # by observation, not computation
        ])
        outdata = wcsmap.tranForward(indata)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(wcsmap, indata)


if __name__ == "__main__":
    unittest.main()
