from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestWcsMap(MappingTestCase):

    def test_WinMap(self):
        winmap = astshim.WinMap([0, 0], [1, 1], [1, 1], [3, 3])
        self.assertIsInstance(winmap, astshim.WinMap)
        self.assertEqual(winmap.getNin(), 2)
        self.assertEqual(winmap.getNout(), 2)

        self.checkBasicSimplify(winmap)
        self.checkCopy(winmap)
        self.checkPersistence(winmap)

        indata = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1, 1],
        ], dtype=float)
        predoutdata = indata * 2 + 1
        outdata = winmap.tranForward(indata)
        self.assertTrue(np.allclose(outdata, predoutdata))

        self.checkRoundTrip(winmap, indata)


if __name__ == "__main__":
    unittest.main()
