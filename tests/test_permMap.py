from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestPermMap(MappingTestCase):

    def test_PermMapMatched(self):
        """Test a PermMap whose inverse is the inverse of its forward
        """
        permmap = astshim.PermMap([2, 3, 1], [3, 1, 2])
        self.assertEqual(permmap.getClass(), "PermMap")
        self.assertEqual(permmap.getNin(), 3)
        self.assertEqual(permmap.getNout(), 3)

        self.checkBasicSimplify(permmap)
        self.checkCopy(permmap)
        self.checkPersistence(permmap)

        indata = np.array([
            [1.1, 2.2, 3.3],
            [-43.5, 1309.31, 0.005],
        ])
        outdata = permmap.tranForward(indata)
        desoutdata = np.array([
            [3.3, 1.1, 2.2],
            [0.005, -43.5, 1309.31],
        ])
        self.assertTrue(np.allclose(outdata, desoutdata))

        self.checkRoundTrip(permmap, indata)

    def test_PermMapUnmatched(self):
        """Test PermMap with different number of inputs and outputs
        """
        permmap = astshim.PermMap([2, 1, 3], [3, 1])
        self.assertEqual(permmap.getClass(), "PermMap")
        self.assertEqual(permmap.getNin(), 3)
        self.assertEqual(permmap.getNout(), 2)

        self.checkPersistence(permmap)

        indata = np.array([1.1, 2.2, -3.3])
        indata.shape = (1, 3)
        outdata = permmap.tranForward(indata)
        self.assertTrue(np.allclose(outdata, [-3.3, 1.1]))

        indata = np.array([1.1, 2.2])
        indata.shape = (1, 2)
        outdata = permmap.tranInverse(indata)
        self.assertTrue(np.allclose(
            outdata, [2.2, 1.1, np.nan], equal_nan=True))

    def test_PermMapWithConstants(self):
        """Test a PermMap with constant values
        """
        permmap = astshim.PermMap([-2, 1, 3], [2, 1, -1], [75.3, -126.5])
        self.assertEqual(permmap.getClass(), "PermMap")
        self.assertEqual(permmap.getNin(), 3)
        self.assertEqual(permmap.getNout(), 3)

        indata = np.array([1.1, 2.2, 3.3])
        indata.shape = (1, 3)
        outdata = permmap.tranForward(indata)
        self.assertTrue(np.allclose(outdata, [2.2, 1.1, 75.3]))

        outdata2 = permmap.tranInverse(indata)
        self.assertTrue(np.allclose(outdata2, [-126.5, 1.1, 3.3]))


if __name__ == "__main__":
    unittest.main()
