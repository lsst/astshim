from __future__ import absolute_import, division, print_function
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestLutMap(MappingTestCase):

    def test_LutMap(self):
        offset = 1.0
        divisor = 0.5
        lutmap = astshim.LutMap([1, 2, 4, 8], offset, divisor)
        self.assertEqual(lutmap.getClass(), "LutMap")
        self.assertEqual(lutmap.getNout(), 1)

        self.checkBasicSimplify(lutmap)
        self.checkCopy(lutmap)
        self.checkPersistence(lutmap)

        indata, desoutdata = zip(*[
            (1.0, 1.0),   # (1 - 1)/0.5 = 0 -> 1
            (1.25, 1.5),  # (1.25 - 1)/0.5 = 0.5 -> 1.5 by interpolation
            (1.5, 2.0),   # (1.5 - 1)/0.5 = 1 -> 2
            (2.0, 4.0),   # (2 - 1)/0.5 = 2 -> 4
            (2.5, 8.0),   # (2.5 - 1)/0.5 = 3 -> 8
        ])
        indata = np.array(indata)
        indata.shape = (len(indata), 1)
        desoutdata = np.array(desoutdata)
        desoutdata.shape = (len(desoutdata), 1)

        outarr = lutmap.tranForward(indata)
        assert_allclose(outarr, desoutdata)
        self.checkRoundTrip(lutmap, indata)

        self.assertEqual(lutmap.getLutInterp(), 0)
        self.assertAlmostEqual(lutmap.getLutEpsilon(),
                               sys.float_info.epsilon, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
