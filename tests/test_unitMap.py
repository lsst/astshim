from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestUnitMap(MappingTestCase):

    def test_UnitMapBasics(self):
        unitmap = astshim.UnitMap(3)
        self.assertEqual(unitmap.getClass(), "UnitMap")
        self.assertEqual(unitmap.getNin(), 3)
        self.assertEqual(unitmap.getNout(), 3)

        self.checkBasicSimplify(unitmap)
        self.checkCast(unitmap, goodType=astshim.Mapping, badType=astshim.ZoomMap)
        self.checkCopy(unitmap)
        self.checkPersistence(unitmap)

        indata = np.array([
            [1.1, 2.2, 3.3],
            [-43.5, 1309.31, 0.005],
        ])
        outdata = unitmap.tran(indata)
        self.assertTrue(np.allclose(outdata, indata))
        self.checkRoundTrip(unitmap, indata)


if __name__ == "__main__":
    unittest.main()
