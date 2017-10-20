from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestWcsMap(MappingTestCase):

    def test_WcsMap(self):
        # Test the Aitoff projection because it is locally flat at 0,0
        # unlike the tangent-plane projection which maps (0, pi/2) -> 0,0
        # and is not very well behaved at nearby points (as expected at a
        # pole).
        wcsmap = ast.WcsMap(2, ast.WcsType.AIT, 1, 2)
        self.assertIsInstance(wcsmap, ast.WcsMap)
        self.assertIsInstance(wcsmap, ast.Mapping)
        self.assertEqual(wcsmap.nIn, 2)
        self.assertEqual(wcsmap.nOut, 2)

        self.assertEqual(wcsmap.natLon, 0)
        self.assertEqual(wcsmap.natLat, 0)
        self.assertEqual(wcsmap.getPVMax(1), 4)
        self.assertEqual(wcsmap.getPVi_m(1, 0), 0)
        self.assertEqual(wcsmap.getPVi_m(1, 1), 0)
        self.assertEqual(wcsmap.getPVi_m(1, 2), 0)
        self.assertGreater(abs(wcsmap.getPVi_m(1, 3)), 1e99)
        self.assertGreater(abs(wcsmap.getPVi_m(1, 4)), 1e99)
        self.assertEqual(wcsmap.getPVMax(2), 0)
        self.assertEqual(wcsmap.wcsAxis, (1, 2))
        self.assertEqual(wcsmap.wcsType, ast.WcsType.AIT)

        self.checkBasicSimplify(wcsmap)
        self.checkCopy(wcsmap)

        indata = np.array([
            [0.0, 0.001, 0.0, 0.0],
            [0.0, 0.0, 0.001, 1.0],
        ], dtype=float)
        pred_outdata = np.array([
            [0.0, 0.001, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.95885108],
        ])
        outdata = wcsmap.applyForward(indata)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(wcsmap, indata)
        self.checkMappingPersistence(wcsmap, indata)


if __name__ == "__main__":
    unittest.main()
