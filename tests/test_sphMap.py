from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestSphMap(MappingTestCase):

    def test_SphMapBasics(self):
        sphmap = astshim.SphMap()
        self.assertEqual(sphmap.getClassName(), "SphMap")
        self.assertEqual(sphmap.getNIn(), 3)
        self.assertEqual(sphmap.getNOut(), 2)
        self.assertEqual(sphmap.getPolarLong(), 0)
        self.assertFalse(sphmap.getUnitRadius())

        self.checkCopy(sphmap)
        self.checkPersistence(sphmap)

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
        outdata = sphmap.tranForward(indata)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(sphmap, indata)

    def test_SphMapAttributes(self):
        sphmap = astshim.SphMap("PolarLong=0.5, UnitRadius=1")
        self.assertEqual(sphmap.getPolarLong(), 0.5)
        self.assertTrue(sphmap.getUnitRadius())


if __name__ == "__main__":
    unittest.main()
