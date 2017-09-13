from __future__ import absolute_import, division, print_function
import sys
import unittest

from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestLutMap(MappingTestCase):

    def test_LutMap(self):
        offset = 1.0
        divisor = 0.5
        lutmap = ast.LutMap([1, 2, 4, 8], offset, divisor)
        self.assertEqual(lutmap.className, "LutMap")
        self.assertEqual(lutmap.nOut, 1)

        self.checkBasicSimplify(lutmap)
        self.checkCopy(lutmap)
        self.checkPersistence(lutmap)

        indata, pred_outdata = zip(*[
            (1.0, 1.0),   # (1 - 1)/0.5 = 0 -> 1
            (1.25, 1.5),  # (1.25 - 1)/0.5 = 0.5 -> 1.5 by interpolation
            (1.5, 2.0),   # (1.5 - 1)/0.5 = 1 -> 2
            (2.0, 4.0),   # (2 - 1)/0.5 = 2 -> 4
            (2.5, 8.0),   # (2.5 - 1)/0.5 = 3 -> 8
        ])
        outdata = lutmap.applyForward(indata)
        assert_allclose(outdata, pred_outdata)
        self.checkRoundTrip(lutmap, indata)

        self.assertEqual(lutmap.lutInterp, 0)
        self.assertAlmostEqual(lutmap.lutEpsilon,
                               sys.float_info.epsilon, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
