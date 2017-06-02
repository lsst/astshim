
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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)
        self.assertTrue(pm.iterInverse)
        self.assertEqual(pm.nIterInverse, 4)
        self.assertAlmostEqual(pm.tolInverse, 1.0E-6)
        self.assertTrue(pm.hasForward)
        self.assertTrue(pm.hasInverse)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        outdata = pm.tranForward(indata)
        xin, yin = indata
        pred_xout = (1.2 * xin * xin) - (0.5 * yin * xin)
        pred_yout = yin
        xout, yout = outdata
        npt.assert_allclose(xout, pred_xout)
        npt.assert_allclose(yout, pred_yout)

        indata_roundtrip = pm.tranInverse(outdata)
        npt.assert_allclose(indata, indata_roundtrip, atol=1.0e-4)

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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)
        self.assertTrue(pm.iterInverse)
        self.assertEqual(pm.nIterInverse, 6)
        self.assertAlmostEqual(pm.tolInverse, 1.2E-7)
        self.assertTrue(pm.hasForward)
        self.assertTrue(pm.hasInverse)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        outdata = pm.tranForward(indata)
        xin, yin = indata
        pred_xout = (1.2 * xin * xin) - (0.5 * yin * xin)
        pred_yout = yin
        xout, yout = outdata
        npt.assert_allclose(xout, pred_xout)
        npt.assert_allclose(yout, pred_yout)

        indata_roundtrip = pm.tranInverse(outdata)
        npt.assert_allclose(indata, indata_roundtrip, atol=1.0e-6)

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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)
        self.assertTrue(pm.hasForward)
        self.assertFalse(pm.hasInverse)
        self.assertFalse(pm.iterInverse)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        outdata = pm.tranForward(indata)
        with self.assertRaises(RuntimeError):
            pm.tranInverse(indata)

        pminv = pm.getInverse()
        self.assertFalse(pminv.hasForward)
        self.assertTrue(pminv.hasInverse)
        self.assertTrue(pminv.isInverted)
        self.assertFalse(pm.iterInverse)

        outdata2 = pminv.tranInverse(indata)
        # outdata and outdata2 should be identical because inverting
        # swaps the behavior of tranForward and tranInverse
        npt.assert_equal(outdata, outdata2)
        with self.assertRaises(RuntimeError):
            pminv.tranForward(indata)

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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        self.checkRoundTrip(pm, indata)

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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        self.assertFalse(pm.hasForward)
        self.assertTrue(pm.hasInverse)
        self.assertFalse(pm.iterInverse)

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
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)
        self.checkPersistence(pm)

        self.assertTrue(pm.hasForward)
        self.assertFalse(pm.hasInverse)
        self.assertFalse(pm.iterInverse)

    def test_PolyMapNoTransform(self):
        """Test constructing a PolyMap with neither forward nor inverse
        coefficients
        """
        coeff_f = np.array([], dtype=float)
        coeff_f.shape = (0, 4)
        coeff_i = np.array([], dtype=float)
        coeff_i.shape = (0, 3)

        with self.assertRaises(ValueError):
            astshim.PolyMap(coeff_f, coeff_i)

        with self.assertRaises(ValueError):
            astshim.PolyMap(coeff_f, 3)

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

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        outdata = pm.tranForward(indata)

        # create a PolyMap with an identical forward transform and a fit inverse
        forward = False
        pm2 = pm.polyTran(forward, 1.0E-10, 1.0E-10, 4, [-1.0, -1.0], [1.0, 1.0])
        outdata2 = pm2.tranForward(indata)
        npt.assert_equal(outdata, outdata2)
        indata2 = pm2.tranInverse(outdata)
        npt.assert_allclose(indata, indata2, atol=1.0e-10)

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

        indata = np.array([-0.5, 0.5, 1.1, 1.8])
        pred_outdata = (2.0*indata.T**2 - indata.T**3).T
        outdata = pm.tranForward(indata)
        npt.assert_allclose(outdata, pred_outdata)

        # the iterative inverse should give valid values
        indata_iterative = pm.tranInverse(outdata)
        outdata_roundtrip = pm.tranForward(indata_iterative)
        npt.assert_allclose(outdata, outdata_roundtrip)

        with self.assertRaises(RuntimeError):
            # includes the range where the inverse has multiple values,
            # so no inverse is possible
            pm.polyTran(False, 1e-3, 1e-3, 10, [-1.0], [2.5])

    def test_PolyMapDM10496(self):
        """Test for a segfault when simplifying a SeriesMap

        We saw an intermittent segfault when simplifying a SeriesMap
        consisting of the inverse of PolyMap with 2 inputs and one output
        followed by its inverse (which should simplify to a UnitMap
        with one input and one output). David Berry fixed this bug in AST
        2017-05-10.

        I tried this test on an older version of astshim and found that it
        triggering a segfault nearly every time.
        """
        coeff_f = np.array([
            [-1.1, 1, 2, 0],
            [1.3, 1, 3, 1],
        ])
        coeff_i = np.array([
            [1.6, 1, 3],
            [-3.6, 2, 1],
        ])

        # execute many times to increase the odds of a segfault
        for i in range(1000):
            amap = astshim.PolyMap(coeff_f, coeff_i)
            amapinv = amap.getInverse()
            cmp2 = amapinv.then(amap)
            result = cmp2.simplify()
            self.assertIsInstance(result, astshim.UnitMap)


if __name__ == "__main__":
    unittest.main()
