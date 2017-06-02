from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestUnitMap(MappingTestCase):

    def test_UnitMapBasics(self):
        unitmap = astshim.UnitMap(3)
        self.assertEqual(unitmap.className, "UnitMap")
        self.assertEqual(unitmap.nIn, 3)
        self.assertEqual(unitmap.nOut, 3)

        self.checkBasicSimplify(unitmap)
        self.checkCopy(unitmap)
        self.checkPersistence(unitmap)

        indata = np.array([
            [1.1, 2.2, 3.3, 4.4],
            [-43.5, 1309.31, 0.005, -36.5],
            [0.0, -2.3, 44.4, 3.14],
        ])
        outdata = unitmap.applyForward(indata)
        assert_allclose(outdata, indata)
        self.checkRoundTrip(unitmap, indata)


if __name__ == "__main__":
    unittest.main()
