import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestShiftMap(MappingTestCase):

    def test_ShiftMapBasics(self):
        offset = np.array([1.1, -2.2, 3.3])
        shiftmap = ast.ShiftMap(offset)
        self.assertEqual(shiftmap.className, "ShiftMap")
        self.assertEqual(shiftmap.nIn, 3)
        self.assertEqual(shiftmap.nOut, 3)

        self.checkBasicSimplify(shiftmap)
        self.checkCopy(shiftmap)

        indata = np.array([
            [1.1, -43.5],
            [2.2, 1309.31],
            [3.3, 0.005],
        ])
        outdata = shiftmap.applyForward(indata)
        pred_outdata = (indata.T + offset).T
        assert_allclose(outdata, pred_outdata)
        self.checkRoundTrip(shiftmap, indata)
        self.checkMappingPersistence(shiftmap, indata)


if __name__ == "__main__":
    unittest.main()
