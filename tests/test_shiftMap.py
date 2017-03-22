from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestShiftMap(MappingTestCase):

    def test_ShiftMapBasics(self):
        offset = np.array([1.1, -2.2, 3.3])
        shiftmap = astshim.ShiftMap(offset)
        self.assertEqual(shiftmap.getClass(), "ShiftMap")
        self.assertEqual(shiftmap.getNin(), 3)
        self.assertEqual(shiftmap.getNout(), 3)

        self.checkBasicSimplify(shiftmap)
        self.checkCopy(shiftmap)
        self.checkPersistence(shiftmap)

        indata = np.array([
            [1.1, 2.2, 3.3],
            [-43.5, 1309.31, 0.005],
        ])
        outdata = shiftmap.tranForward(indata)
        assert_allclose(outdata, indata + offset)
        self.checkRoundTrip(shiftmap, indata)


if __name__ == "__main__":
    unittest.main()
