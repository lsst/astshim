from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestPermMap(MappingTestCase):

    def test_PermMapMatched(self):
        """Test a PermMap whose inverse is the inverse of its forward
        """
        permmap = astshim.PermMap([2, 3, 1], [3, 1, 2])
        self.assertEqual(permmap.className, "PermMap")
        self.assertEqual(permmap.nIn, 3)
        self.assertEqual(permmap.nOut, 3)

        self.checkBasicSimplify(permmap)
        self.checkCopy(permmap)
        self.checkPersistence(permmap)

        indata = np.array([
            [1.1, -43.5],
            [2.2, 1309.31],
            [3.3, 0.005],
        ])
        outdata = permmap.applyForward(indata)
        pred_outdata = np.array([
            [3.3, 0.005],
            [1.1, -43.5],
            [2.2, 1309.31],
        ])
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(permmap, indata)

    def test_PermMapUnmatched(self):
        """Test PermMap with different number of inputs and outputs
        """
        permmap = astshim.PermMap([2, 1, 3], [3, 1])
        self.assertEqual(permmap.className, "PermMap")
        self.assertEqual(permmap.nIn, 3)
        self.assertEqual(permmap.nOut, 2)

        self.checkPersistence(permmap)

        indata = np.array([1.1, 2.2, -3.3])
        outdata = permmap.applyForward(indata)
        assert_allclose(outdata, [-3.3, 1.1])

        indata = np.array([1.1, 2.2])
        outdata = permmap.applyInverse(indata)
        assert_allclose(outdata, [2.2, 1.1, np.nan], equal_nan=True)

    def test_PermMapWithConstants(self):
        """Test a PermMap with constant values
        """
        permmap = astshim.PermMap([-2, 1, 3], [2, 1, -1], [75.3, -126.5])
        self.assertEqual(permmap.className, "PermMap")
        self.assertEqual(permmap.nIn, 3)
        self.assertEqual(permmap.nOut, 3)

        indata = np.array([1.1, 2.2, 3.3])
        outdata = permmap.applyForward(indata)
        assert_allclose(outdata, [2.2, 1.1, 75.3])

        outdata2 = permmap.applyInverse(indata)
        assert_allclose(outdata2, [-126.5, 1.1, 3.3])


if __name__ == "__main__":
    unittest.main()
