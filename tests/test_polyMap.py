from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestMatrixMap(MappingTestCase):

    def test_PolyMapUnidirectional(self):
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, 2)
        self.assertIsInstance(pm, astshim.Object)
        self.assertIsInstance(pm, astshim.Mapping)
        self.assertIsInstance(pm, astshim.PolyMap)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)
        self.assertEqual(pm.getNiterInverse(), 4)
        self.assertAlmostEqual(pm.getTolInverse(), 1.0E-6)

        # checkBasicSimplify segfaults!
        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        pout = pm.tranForward(pin)
        for (xi, yi, xo, yo) in zip(pin[:, 0], pin[:, 1], pout[:, 0], pout[:, 1]):
            xn = 1.2 * xi * xi - 0.5 * yi * xi
            yn = yi
            self.assertAlmostEqual(xn, xo)
            self.assertAlmostEqual(yn, yo)

    def test_PolyMapBidirectional(self):
        coeff_f = np.array([
            [1., 1, 1, 0],
            [1., 1, 0, 1],
            [1., 2, 1, 0],
            [-1., 2, 0, 1]
        ])
        coeff_i = np.array([
            [0.5, 1, 1, 0],
            [0.5, 1, 0, 1],
            [0.5, 2, 1, 0],
            [-0.5, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, coeff_i)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])

        self.checkRoundTrip(pm, pin)

        new = pm.polyTran(False, 1.0E-8, 0.01, 2, [-1.0, -1.0], [1.0, 1.0])
        pout = new.tranForward(pin)
        pnew = new.tranInverse(pout)
        for (xi, yi, xn, yn) in zip(pin[:, 0], pin[:, 1], pnew[:, 0], pnew[:, 1]):
            self.assertAlmostEqual(xn, xi)
            self.assertAlmostEqual(yn, yi)

    def test_PolyMapPolyTran(self):
        coeff_f = np.array([
            [1., 1, 1, 0],
            [1., 1, 0, 1],
            [1., 2, 1, 0],
            [-1., 2, 0, 1]
        ])
        coeff_i = np.array([
            [0.5, 1, 1, 0],
            [0.5, 1, 0, 1],
            [0.5, 2, 1, 0],
            [-0.5, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, coeff_i)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])

        new = pm.polyTran(False, 1.0E-8, 0.01, 2, [-1.0, -1.0], [1.0, 1.0])
        pout = new.tranForward(pin)
        pnew = new.tranInverse(pout)
        for (xi, yi, xn, yn) in zip(pin[:, 0], pin[:, 1], pnew[:, 0], pnew[:, 1]):
            self.assertAlmostEqual(xn, xi)
            self.assertAlmostEqual(yn, yi)


if __name__ == "__main__":
    unittest.main()
