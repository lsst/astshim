from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestSphMap(MappingTestCase):

    def test_SphMapBasics(self):
        sphmap = ast.SphMap()
        self.assertEqual(sphmap.className, "SphMap")
        self.assertEqual(sphmap.nIn, 3)
        self.assertEqual(sphmap.nOut, 2)
        self.assertEqual(sphmap.polarLong, 0)
        self.assertFalse(sphmap.unitRadius)

        self.checkCopy(sphmap)
        # SphMap followed by an inverse, simplified, is a compound map,
        # not a UnitMap, since data only round trips for unit vectors.
        # Hence the following test instead of checkBasicSimplify:
        simplified = sphmap.then(sphmap.inverted()).simplify()
        self.assertTrue(isinstance(simplified, ast.CmpMap))

        # for data to round trip, all inputs must be unit vectors
        indata = np.array([
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0 / math.sqrt(3.0)],
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0 / math.sqrt(3.0)],
            [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0 / math.sqrt(3.0)],
        ], dtype=float)
        halfpi = math.pi / 2.0
        pred_outdata = np.array([
            [0.0, halfpi, 0.0, math.pi, -halfpi, 0.0, math.pi / 4.0],
            [0.0, 0.0, halfpi, 0.0, 0.0, -halfpi, math.atan(1.0 / math.sqrt(2.0))],
        ], dtype=float)
        outdata = sphmap.applyForward(indata)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(sphmap, indata)
        self.checkMappingPersistence(sphmap, indata)

    def test_SphMapAttributes(self):
        sphmap = ast.SphMap("PolarLong=0.5, UnitRadius=1")
        self.assertEqual(sphmap.polarLong, 0.5)
        self.assertTrue(sphmap.unitRadius)


if __name__ == "__main__":
    unittest.main()
