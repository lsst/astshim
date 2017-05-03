
from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
import numpy.testing as npt

import astshim
from astshim.test import MappingTestCase


class TestPolyMap(MappingTestCase):

    def test_PolyMapIterativeInverse(self):
        """Test a unidirectional polymap with its default iterative inverse
        """
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, 2, "IterInverse=1")
        self.assertIsInstance(pm, astshim.Object)
        self.assertIsInstance(pm, astshim.Mapping)
        self.assertIsInstance(pm, astshim.PolyMap)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)
        self.assertEqual(pm.getNiterInverse(), 4)
        self.assertAlmostEqual(pm.getTolInverse(), 1.0E-6)
        self.assertTrue(pm.hasForward())
        self.assertTrue(pm.hasInverse())

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        pout = pm.tranForward(pin)
        xin, yin = pin.transpose()
        xoutExpected = (1.2 * xin * xin) - (0.5 * yin * xin)
        youtExpected = yin
        xout, yout = pout.transpose()
        npt.assert_allclose(xout, xoutExpected)
        npt.assert_allclose(yout, youtExpected)

        pinRoundTrip = pm.tranInverse(pout)
        npt.assert_allclose(pin, pinRoundTrip, atol=1.0e-4)

    def test_polyMapAtributes(self):
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, 2, "IterInverse=1, NIterInverse=6, TolInverse=1.2e-7")
        self.assertIsInstance(pm, astshim.Object)
        self.assertIsInstance(pm, astshim.Mapping)
        self.assertIsInstance(pm, astshim.PolyMap)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)
        self.assertEqual(pm.getNiterInverse(), 6)
        self.assertAlmostEqual(pm.getTolInverse(), 1.2E-7)
        self.assertTrue(pm.hasForward())
        self.assertTrue(pm.hasInverse())

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        pout = pm.tranForward(pin)
        xin, yin = pin.transpose()
        xoutExpected = (1.2 * xin * xin) - (0.5 * yin * xin)
        youtExpected = yin
        xout, yout = pout.transpose()
        npt.assert_allclose(xout, xoutExpected)
        npt.assert_allclose(yout, youtExpected)

        pinRoundTrip = pm.tranInverse(pout)
        npt.assert_allclose(pin, pinRoundTrip, atol=1.0e-6)

    def test_polyMapNoInverse(self):
        """Test a unidirectional polymap with no numeric inverse
        """
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, 2)
        self.assertIsInstance(pm, astshim.PolyMap)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)
        self.assertTrue(pm.hasForward())
        self.assertFalse(pm.hasInverse())

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ])
        pout = pm.tranForward(pin)
        with self.assertRaises(RuntimeError):
            pm.tranInverse(pin)

        pminv = pm.getInverse()
        self.assertFalse(pminv.hasForward())
        self.assertTrue(pminv.hasInverse())
        self.assertTrue(pminv.isInverted())

        pout2 = pminv.tranInverse(pin)
        # pout and pout2 should be identical because inverting
        # swaps the behavior of tranForward and tranInverse
        npt.assert_equal(pout, pout2)
        with self.assertRaises(RuntimeError):
            pminv.tranForward(pin)

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

    def test_PolyMapEmptyForwardCoeffs(self):
        """Test constructing a PolyMap with empty forward coefficients
        """
        coeff_f = np.array([], dtype=float)
        coeff_f.shape = (0, 4)
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

        self.assertFalse(pm.hasForward())
        self.assertTrue(pm.hasInverse())

    def test_PolyMapEmptyInverseCoeffs(self):
        """Test constructing a PolyMap with empty inverse coefficients
        """
        coeff_f = np.array([
            [1., 1, 1, 0],
            [1., 1, 0, 1],
            [1., 2, 1, 0],
            [-1., 2, 0, 1]
        ])
        coeff_i = np.array([], dtype=float)
        coeff_i.shape = (0, 4)
        pm = astshim.PolyMap(coeff_f, coeff_i)
        self.assertEqual(pm.getNin(), 2)
        self.assertEqual(pm.getNout(), 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        self.assertTrue(pm.hasForward())
        self.assertFalse(pm.hasInverse())

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

        pout = pm.tranForward(pin)

        # create a PolyMap with an identical forward transform and a fit inverse
        forward = False
        pm2 = pm.polyTran(forward, 1.0E-10, 1.0E-10, 4, [-1.0, -1.0], [1.0, 1.0])
        pout2 = pm2.tranForward(pin)
        npt.assert_equal(pout, pout2)
        pin2 = pm2.tranInverse(pout)
        npt.assert_allclose(pin, pin2, atol=1.0e-10)

    def test_PolyMapPolyMapUnivertible(self):
        """Test polyTran on a PolyMap without a single-valued inverse

        The equation is y = x^2 - x^3, whose inverse has 3 values
        between roughly -0.66 and 2.0
        """
        coeff_f = np.array([
            [2.0, 1, 2],
            [-1.0, 1, 3],
        ])
        pm = astshim.PolyMap(coeff_f, 1, "IterInverse=1")

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        pin = np.array([
            [-0.5],
            [0.5],
            [1.1],
            [1.8],
        ])
        des_pout = 2.0*pin**2 - pin**3
        pout = pm.tranForward(pin)
        npt.assert_allclose(pout, des_pout)

        # the iterative inverse should give valid values
        pinIterative = pm.tranInverse(pout)
        poutRoundTrip = pm.tranForward(pinIterative)
        npt.assert_allclose(pout, poutRoundTrip)

        with self.assertRaises(RuntimeError):
            # includes the range where the inverse has multiple values,
            # so no inverse is possible
            pm.polyTran(False, 1e-3, 1e-3, 10, [-1.0], [2.5])


if __name__ == "__main__":
    unittest.main()
