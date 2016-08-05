from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestSphMap(MappingTestCase):

    def test_SphMapBasics(self):
        sphmap = astshim.SphMap()
        self.assertEqual(sphmap.getClass(), "SphMap")
        self.assertEqual(sphmap.getNin(), 3)
        self.assertEqual(sphmap.getNout(), 2)
        self.assertEqual(sphmap.getPolarLong(), 0)
        self.assertFalse(sphmap.getUnitRadius())

        self.checkCast(sphmap, goodType=astshim.Mapping, badType=astshim.ZoomMap)
        self.checkCopy(sphmap)
        self.checkPersistence(sphmap)

        indata = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)],  # normalize to round trip works
        ], dtype=float)
        predoutdata = np.array([
            [0, 0],
            [math.pi/2, 0],
            [0, math.pi/2],
            [math.pi, 0],
            [-math.pi/2, 0],
            [0, -math.pi/2],
            [math.pi/4, math.atan(1/math.sqrt(2))],
        ], dtype=float)
        outdata = sphmap.tran(indata)
        self.assertTrue(np.allclose(outdata, predoutdata))

        self.checkRoundTrip(sphmap, indata)

    def test_SphMapAttributes(self):
        sphmap = astshim.SphMap("PolarLong=0.5, UnitRadius=1")
        self.assertEqual(sphmap.getPolarLong(), 0.5)
        self.assertTrue(sphmap.getUnitRadius())


if __name__ == "__main__":
    unittest.main()
