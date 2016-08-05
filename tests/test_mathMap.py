from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestMathMap(MappingTestCase):

    def test_MathMapInvertible(self):
        mathmap = astshim.MathMap(2, 2,
                                  ["r = sqrt(x * x + y * y)",
                                   "theta = atan2(y, x)"],
                                  ["x = r * cos(theta)", "y = r * sin(theta)"],
                                  "SimpIF=1, SimpFI=1, Seed=-57")
        self.assertEqual(mathmap.getClass(), "MathMap")
        self.assertEqual(mathmap.getNin(), 2)
        self.assertEqual(mathmap.getNout(), 2)

        self.checkBasicSimplify(mathmap)
        self.checkCopy(mathmap)
        self.checkPersistence(mathmap)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        pout = mathmap.tranForward(pin)
        x = pin[:, 0]
        y = pin[:, 1]
        r = pout[:, 0]
        theta = pout[:, 1]
        desR = np.sqrt(x * x + y * y)
        desTheta = np.arctan2(y, x)
        self.assertTrue(np.allclose(r, desR))
        self.assertTrue(np.allclose(theta, desTheta))

        self.checkRoundTrip(mathmap, pin)

        self.assertEqual(mathmap.getSeed(), -57)
        self.assertTrue(mathmap.getSimpFI())
        self.assertTrue(mathmap.getSimpIF())

    def test_MathMapNonInvertible(self):
        mathmap = astshim.MathMap(2, 1,
                                  ["r = sqrt(x * x + y * y)"],
                                  ["x = r", "y = 0"])
        self.assertEqual(mathmap.getClass(), "MathMap")
        self.assertEqual(mathmap.getNin(), 2)
        self.assertEqual(mathmap.getNout(), 1)

        self.checkPersistence(mathmap)
        with self.assertRaises(AssertionError):
            self.checkBasicSimplify(mathmap)

        self.assertFalse(mathmap.getSimpFI())
        self.assertFalse(mathmap.getSimpIF())


if __name__ == "__main__":
    unittest.main()
