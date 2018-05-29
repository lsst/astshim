
from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
import numpy.testing as npt

import astshim as ast
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
        pm = ast.PolyMap(coeff_f, 2, "IterInverse=1")
        self.assertIsInstance(pm, ast.Object)
        self.assertIsInstance(pm, ast.Mapping)
        self.assertIsInstance(pm, ast.PolyMap)
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)
        self.assertTrue(pm.iterInverse)
        self.assertEqual(pm.nIterInverse, 4)
        self.assertAlmostEqual(pm.tolInverse, 1.0E-6)
        self.assertTrue(pm.hasForward)
        self.assertTrue(pm.hasInverse)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        outdata = pm.applyForward(indata)
        xin, yin = indata
        pred_xout = (1.2 * xin * xin) - (0.5 * yin * xin)
        pred_yout = yin
        xout, yout = outdata
        npt.assert_allclose(xout, pred_xout, atol=1e-12)
        npt.assert_allclose(yout, pred_yout, atol=1e-12)

        indata_roundtrip = pm.applyInverse(outdata)
        npt.assert_allclose(indata, indata_roundtrip, atol=1.0e-4)

        self.checkMappingPersistence(pm, indata)

    def test_polyMapAttributes(self):
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = ast.PolyMap(coeff_f, 2, "IterInverse=1, NIterInverse=6, TolInverse=1.2e-7")
        self.assertIsInstance(pm, ast.Object)
        self.assertIsInstance(pm, ast.Mapping)
        self.assertIsInstance(pm, ast.PolyMap)
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
        outdata = pm.applyForward(indata)
        xin, yin = indata
        pred_xout = (1.2 * xin * xin) - (0.5 * yin * xin)
        pred_yout = yin
        xout, yout = outdata
        npt.assert_allclose(xout, pred_xout, atol=1e-12)
        npt.assert_allclose(yout, pred_yout, atol=1e-12)

        indata_roundtrip = pm.applyInverse(outdata)
        npt.assert_allclose(indata, indata_roundtrip, atol=1.0e-6)

        self.checkMappingPersistence(pm, indata)

    def test_polyMapNoInverse(self):
        """Test a unidirectional polymap with no numeric inverse
        """
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = ast.PolyMap(coeff_f, 2)
        self.assertIsInstance(pm, ast.PolyMap)
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)
        self.assertTrue(pm.hasForward)
        self.assertFalse(pm.hasInverse)
        self.assertFalse(pm.iterInverse)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        outdata = pm.applyForward(indata)
        with self.assertRaises(RuntimeError):
            pm.applyInverse(indata)

        pminv = pm.getInverse()
        self.assertFalse(pminv.hasForward)
        self.assertTrue(pminv.hasInverse)
        self.assertTrue(pminv.isInverted)
        self.assertFalse(pm.iterInverse)

        outdata2 = pminv.applyInverse(indata)
        # outdata and outdata2 should be identical because inverting
        # swaps the behavior of applyForward and applyInverse
        npt.assert_equal(outdata, outdata2)
        with self.assertRaises(RuntimeError):
            pminv.applyForward(indata)

        self.checkMappingPersistence(pm, indata)

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
        pm = ast.PolyMap(coeff_f, coeff_i)
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        self.checkRoundTrip(pm, indata)
        self.checkMappingPersistence(pm, indata)

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
        pm = ast.PolyMap(coeff_f, coeff_i)
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)

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
        pm = ast.PolyMap(coeff_f, coeff_i)
        self.assertEqual(pm.nIn, 2)
        self.assertEqual(pm.nOut, 2)

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)

        self.assertTrue(pm.hasForward)
        self.assertFalse(pm.hasInverse)
        self.assertFalse(pm.iterInverse)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])
        self.checkMappingPersistence(pm, indata)

    def test_PolyMapNoTransform(self):
        """Test constructing a PolyMap with neither forward nor inverse
        coefficients
        """
        coeff_f = np.array([], dtype=float)
        coeff_f.shape = (0, 4)
        coeff_i = np.array([], dtype=float)
        coeff_i.shape = (0, 3)

        with self.assertRaises(ValueError):
            ast.PolyMap(coeff_f, coeff_i)

        with self.assertRaises(ValueError):
            ast.PolyMap(coeff_f, 3)

    def test_PolyMapPolyTranTrivial(self):
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
        pm = ast.PolyMap(coeff_f, coeff_i)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        outdata = pm.applyForward(indata)

        # create a PolyMap with an identical forward transform and a fit inverse
        forward = False
        pm2 = pm.polyTran(forward, 1.0E-10, 1.0E-10, 4, [-1.0, -1.0], [1.0, 1.0])
        outdata2 = pm2.applyForward(indata)
        npt.assert_equal(outdata, outdata2)
        indata2 = pm2.applyInverse(outdata)
        npt.assert_allclose(indata, indata2, atol=1.0e-12)

        self.checkMappingPersistence(pm, indata)
        self.checkMappingPersistence(pm2, indata)

    def test_PolyMapPolyTranNontrivial(self):
        """Test PolyMap.polyTran on a non-trivial case
        """
        # Approximate "field angle to focal plane" transformation coefficients for LSST
        # thus the domain of the forward direction is 1.75 degrees = 0.0305 radians
        # The camera has 10 um pixels = 0.01 mm
        # The desired accuracy of the inverse transformation is
        # 0.001 pixels = 1e-5 mm = 9.69e-10 radians
        plateScaleRad = 9.69627362219072e-05  # radians per mm
        radialCoeff = np.array([0.0, 1.0, 0.0, 0.925]) / plateScaleRad
        polyCoeffs = []
        for i, coeff in enumerate(radialCoeff):
            polyCoeffs.append((coeff, 1, i))
        polyCoeffs = np.array(polyCoeffs)
        fieldAngleToFocalPlane = ast.PolyMap(polyCoeffs, 1)

        atolRad = 1.0e-9
        fieldAngleToFocalPlane2 = fieldAngleToFocalPlane.polyTran(forward=False, acc=atolRad, maxacc=atolRad,
                                                                  maxorder=10, lbnd=[0], ubnd=[0.0305])
        fieldAngle = np.linspace(0, 0.0305, 100)
        focalPlane = fieldAngleToFocalPlane.applyForward(fieldAngle)
        fieldAngleRoundTrip = fieldAngleToFocalPlane2.applyInverse(focalPlane)
        npt.assert_allclose(fieldAngle, fieldAngleRoundTrip, atol=atolRad)

        # Verify that polyTran cannot fit the inverse when maxorder is too small
        with self.assertRaises(RuntimeError):
            fieldAngleToFocalPlane.polyTran(forward=False, acc=atolRad, maxacc=atolRad,
                                            maxorder=3, lbnd=[0], ubnd=[0.0305])

    def test_PolyMapIterInverseDominates(self):
        """Test that IterInverse dominates inverse coefficients for applyInverse
        """
        coeff_f = np.array([
            [1., 1, 1],
        ])
        # these coefficients don't match coeff_f, in that the inverse mapping
        # does not undo the forward mapping (as proven below)
        coeff_i = np.array([
            [25., 1, 2],
        ])
        polyMap = ast.PolyMap(coeff_f, coeff_i, "IterInverse=1")

        indata = np.array([-0.5, 0.5, 1.1, 1.8])
        outdata = polyMap.applyForward(indata)
        indata_roundtrip = polyMap.applyInverse(outdata)
        npt.assert_allclose(indata, indata_roundtrip, atol=1e-12)

        # prove that without the iterative inverse the PolyMap does not invert correctly
        polyMap2 = ast.PolyMap(coeff_f, coeff_i)
        indata_roundtrip2 = polyMap2.applyInverse(outdata)
        self.assertFalse(np.allclose(indata, indata_roundtrip2))

    def test_PolyMapPolyTranIterInverse(self):
        """Test PolyTran on a PolyMap that has an iterative inverse

        The result should use the fit inverse, not the iterative inverse
        """
        coeff_f = np.array([
            [1., 1, 1],
        ])
        for polyMap in (
            ast.PolyMap(coeff_f, 1, "IterInverse=1"),
            ast.PolyMap(coeff_f, coeff_f, "IterInverse=1"),
        ):
            # make sure IterInverse is True and set
            self.assertTrue(polyMap.iterInverse)
            self.assertTrue(polyMap.test("IterInverse"))

            # fit inverse; this should clear iterInverse
            polyMapFitInv = polyMap.polyTran(False, 1.0E-10, 1.0E-10, 4, [-1.0], [1.0])
            self.assertFalse(polyMapFitInv.iterInverse)
            self.assertFalse(polyMapFitInv.test("IterInverse"))

            # fit forward direction of inverted mapping; this should also clear IterInverse
            polyMapInvFitFwd = polyMap.getInverse().polyTran(True, 1.0E-10, 1.0E-10, 4, [-1.0], [1.0])
            self.assertFalse(polyMapInvFitFwd.iterInverse)
            self.assertFalse(polyMapFitInv.test("IterInverse"))

            # cannot fit forward because inverse is iterative
            with self.assertRaises(ValueError):
                polyMap.polyTran(True, 1.0E-10, 1.0E-10, 4, [-1.0], [1.0])

            # cannot fit inverse of inverted mapping because forward is iterative
            with self.assertRaises(ValueError):
                polyMap.getInverse().polyTran(False, 1.0E-10, 1.0E-10, 4, [-1.0], [1.0])

    def test_PolyMapPolyMapUnivertible(self):
        """Test polyTran on a PolyMap without a single-valued inverse

        The equation is y = x^2 - x^3, whose inverse has 3 values
        between roughly -0.66 and 2.0
        """
        coeff_f = np.array([
            [2.0, 1, 2],
            [-1.0, 1, 3],
        ])
        pm = ast.PolyMap(coeff_f, 1, "IterInverse=1")

        self.checkBasicSimplify(pm)
        self.checkCopy(pm)

        indata = np.array([-0.5, 0.5, 1.1, 1.8])
        pred_outdata = (2.0*indata.T**2 - indata.T**3).T
        outdata = pm.applyForward(indata)
        npt.assert_allclose(outdata, pred_outdata, atol=1e-12)

        # the iterative inverse should give valid values
        indata_iterative = pm.applyInverse(outdata)
        outdata_roundtrip = pm.applyForward(indata_iterative)
        npt.assert_allclose(outdata, outdata_roundtrip, atol=1e-12)

        self.checkMappingPersistence(pm, indata)

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
            amap = ast.PolyMap(coeff_f, coeff_i)
            amapinv = amap.getInverse()
            cmp2 = amapinv.then(amap)
            result = cmp2.simplify()
            self.assertIsInstance(result, ast.UnitMap)


if __name__ == "__main__":
    unittest.main()
