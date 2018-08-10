from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebval2d
import numpy.testing as npt

import astshim as ast
from astshim.test import MappingTestCase


def normalize(inArray, lbnd, ubnd):
    """Return the value of x normalized to [-1, 1]

    This is a linear scaling with no bounds checking,
    so if an input value is less than lbnd or greater than ubnd,
    the returned value will be less than -1 or greater than 1

    Parameters
    ----------
    inArray : `numpy.array` of float
        Value(s) to normalize; a list of nAxes x nPoints values
        (the form used by ast.Mapping.applyForward)
    lbnd : sequence of `float`
        Lower bounds (one element per axis)
    ubnd : sequence of `float`
        Upper bounds (one element per axis)

    Returns
    -------
    `numpy.array` of float
        Each value is scaled such to -1 if x = lbnd, 1 if x = ubnd
    """
    # normalize x in the range [-1, 1]
    lbnd = np.array(lbnd)
    ubnd = np.array(ubnd)
    delta = ubnd - lbnd
    return (-1 + ((inArray.T - lbnd) * 2.0 / delta)).T


class ReferenceCheby(object):

    def __init__(self, referenceCheby, lbnd, ubnd):
        """Construct a reference Chebyshev polynomial

        Parameters
        ----------
        referenceCheby : callable
            A function that takes a normalized point (as a list of floats)
            that has been normalized to the range [-1, 1]
            and returns the expected results from ChebyPoly.applyForward
            or applyInverse for the corresponding un-normalized point
        lbnd : list of float
            Lower bounds of inputs (for normalization)
        ubnd : list of float
            Upper bounds of inputs (for normalization)
        """
        self.referenceCheby = referenceCheby
        self.lbnd = lbnd
        self.ubnd = ubnd

    def transform(self, inArray):
        """Transform data using the reference function

        Parameters
        ----------
        inArray : `numpy.array`
            Input array of points in the form used by ChebyMap.applyForward
            or applyInverse.

        Returns
        -------
        outArray : `numpy.array`
            inArray transformed by referenceCheby (after normalizing inArray)
        """
        inNormalized = normalize(inArray, self.lbnd, self.ubnd)
        outdata = [self.referenceCheby(inPoint) for inPoint in inNormalized.T]
        arr = np.array(outdata)
        if len(arr.shape) > 2:
            # trim unwanted extra dimension (occurs when nin=1)
            arr.shape = arr.shape[0:2]
        return arr.T


class TestChebyMap(MappingTestCase):

    def setUp(self):
        self.normErr = "Invalid {0} normalization: min={1}, max={2}, min/max norm=({3}, {4}) != (-1, 1)"
        # We need a slightly larger than the full floating point tolerance for
        # many of these tests.
        self.atol = 5e-14

    def test_chebyMapUnidirectional_2_2(self):
        """Test one-directional ChebyMap with 2 inputs and 2 outputs

        This is a long test because it is a bit of a nuisance setting up
        the reference transform, so once I have it, I use it for three
        different ChebyMaps (forward-only, forward with no inverse,
        and inverse with no forward).
        """
        nin = 2
        nout = 2
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, 2.5]
        # Coefficients for the following polynomial:
        # y1 = 1.2 T2(x1') T0(x2') - 0.5 T1(x1') T1(x2')
        # y2 = 1.0 T0(x1') T1(x2')
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        self.assertEqual(nin, coeff_f.shape[1] - 2)

        def referenceFunc(point):
            """Reference implementation; point must be in range [-1, 1]
            """
            c1 = np.zeros((3, 3))
            c1[2, 0] = 1.2
            c1[1, 1] = -0.5
            c2 = np.zeros((3, 3))
            c2[0, 1] = 1.0
            x1, x2 = point
            return (
                chebval2d(x1, x2, c1),
                chebval2d(x1, x2, c2),
            )

        null_coeff = np.zeros(shape=(0, 4))
        self.assertEqual(nin, null_coeff.shape[1] - 2)

        # arbitary input points that cover the full domain
        indata = np.array([
            [-2.0, -0.5, 0.5, 1.5],
            [-2.5, 1.5, -0.5, 2.5],
        ])

        refCheby = ReferenceCheby(referenceFunc, lbnd_f, ubnd_f)

        # forward-only constructor
        chebyMap1 = ast.ChebyMap(coeff_f, nout, lbnd_f, ubnd_f)
        self.assertIsInstance(chebyMap1, ast.Object)
        self.assertIsInstance(chebyMap1, ast.Mapping)
        self.assertIsInstance(chebyMap1, ast.ChebyMap)
        self.assertEqual(chebyMap1.nIn, nin)
        self.assertEqual(chebyMap1.nOut, nout)
        self.assertTrue(chebyMap1.hasForward)
        self.assertFalse(chebyMap1.hasInverse)
        self.checkBasicSimplify(chebyMap1)
        self.checkCopy(chebyMap1)
        self.checkMappingPersistence(chebyMap1, indata)
        domain1 = chebyMap1.getDomain(forward=True)
        npt.assert_allclose(domain1.lbnd, lbnd_f)
        npt.assert_allclose(domain1.ubnd, ubnd_f)

        outdata = chebyMap1.applyForward(indata)

        with self.assertRaises(RuntimeError):
            chebyMap1.applyInverse(indata)

        pred_outdata = refCheby.transform(indata)
        npt.assert_allclose(outdata, pred_outdata)

        # bidirectional constructor, forward only specified
        chebyMap2 = ast.ChebyMap(coeff_f, null_coeff, lbnd_f, ubnd_f, [], [])
        self.assertIsInstance(chebyMap2, ast.Object)
        self.assertIsInstance(chebyMap2, ast.Mapping)
        self.assertIsInstance(chebyMap2, ast.ChebyMap)
        self.assertEqual(chebyMap2.nIn, nin)
        self.assertEqual(chebyMap2.nOut, nout)
        self.assertTrue(chebyMap2.hasForward)
        self.assertFalse(chebyMap2.hasInverse)
        self.checkBasicSimplify(chebyMap2)
        self.checkCopy(chebyMap2)
        self.checkMappingPersistence(chebyMap1, indata)
        domain2 = chebyMap2.getDomain(forward=True)
        npt.assert_allclose(domain2.lbnd, lbnd_f)
        npt.assert_allclose(domain2.ubnd, ubnd_f)

        outdata2 = chebyMap2.applyForward(indata)
        npt.assert_allclose(outdata2, outdata)

        with self.assertRaises(RuntimeError):
            chebyMap2.applyInverse(indata)

        # bidirectional constructor, inverse only specified
        chebyMap3 = ast.ChebyMap(null_coeff, coeff_f, [], [], lbnd_f, ubnd_f)
        self.assertIsInstance(chebyMap3, ast.Object)
        self.assertIsInstance(chebyMap3, ast.Mapping)
        self.assertIsInstance(chebyMap3, ast.ChebyMap)
        self.assertEqual(chebyMap3.nIn, nin)
        self.assertEqual(chebyMap3.nOut, nout)
        self.assertFalse(chebyMap3.hasForward)
        self.assertTrue(chebyMap3.hasInverse)
        domain3 = chebyMap3.getDomain(forward=False)
        npt.assert_allclose(domain3.lbnd, lbnd_f)
        npt.assert_allclose(domain3.ubnd, ubnd_f)

        outdata3 = chebyMap3.applyInverse(indata)
        npt.assert_allclose(outdata3, outdata)

        with self.assertRaises(RuntimeError):
            chebyMap3.applyForward(indata)

    def test_ChebyMapBidirectional(self):
        """Test a ChebyMap with separate forward and inverse mappings

        For simplicity, they are not the inverse of each other.
        """
        nin = 2
        nout = 1
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, -0.5]

        # cover the domain
        indata_f = np.array([
            [-2.0, -1.5, 0.1, 1.5],
            [-1.0, -2.5, -0.5, -0.5],
        ])

        lbnd_i = [-3.0]
        ubnd_i = [-1.0]

        # cover the domain
        indata_i = np.array([
            [-3.0, -1.1, -1.5, -2.3, -1.0],
        ])
        # Coefficients for the following polynomial:
        # y1 = -1.1 T2(x1') T0(x2') + 1.3 T3(x1') T1(x2')
        coeff_f = np.array([
            [-1.1, 1, 2, 0],
            [1.3, 1, 3, 1],
        ])
        self.assertEqual(nin, coeff_f.shape[1] - 2)

        def referenceFunc_f(point):
            """Reference forward implementation; point must be in range [-1, 1]
            """
            c1 = np.zeros((4, 4))
            c1[2, 0] = -1.1
            c1[3, 1] = 1.3
            x1, x2 = point
            return (
                chebval2d(x1, x2, c1),
            )

        # Coefficients for the following polynomial:
        # y1 = 1.6 T3(x1')
        # y2 = -3.6 T1(x1')
        coeff_i = np.array([
            [1.6, 1, 3],
            [-3.6, 2, 1],
        ])
        self.assertEqual(nout, coeff_i.shape[1] - 2)

        def referenceFunc_i(point):
            """Reference inverse implementation; point must be in range [-1, 1]
            """
            c1 = np.array([0, 0, 0, 1.6], dtype=float)
            c2 = np.array([0, -3.6], dtype=float)
            x1 = point
            return (
                chebval(x1, c1),
                chebval(x1, c2),
            )

        refCheby_f = ReferenceCheby(referenceFunc_f, lbnd_f, ubnd_f)
        refCheby_i = ReferenceCheby(referenceFunc_i, lbnd_i, ubnd_i)

        chebyMap = ast.ChebyMap(coeff_f, coeff_i, lbnd_f, ubnd_f, lbnd_i, ubnd_i)
        self.assertEqual(chebyMap.nIn, 2)
        self.assertEqual(chebyMap.nOut, 1)

        self.checkBasicSimplify(chebyMap)
        self.checkCopy(chebyMap)
        self.checkMappingPersistence(chebyMap, indata_f)

        outdata_f = chebyMap.applyForward(indata_f)
        des_outdata_f = refCheby_f.transform(indata_f)

        npt.assert_allclose(outdata_f, des_outdata_f)

        outdata_i = chebyMap.applyInverse(indata_i)
        des_outdata_i = refCheby_i.transform(indata_i)

        npt.assert_allclose(outdata_i, des_outdata_i)

    def test_ChebyMapPolyTran(self):
        nin = 2
        nout = 2
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, 2.5]

        # arbitrary points that cover the input range
        indata = np.array([
            [-2.0, -1.0, 0.1, 1.5, 1.0],
            [0.0, -2.5, -0.2, 2.5, 2.5],
        ])

        # Coefficients for the following gently varying polynomial:
        # y1 = -2.0 T0(x1') T0(x2') + 0.11 T1(x1') T0(x2') - 0.2 T0(x1') T1(x2') + 0.001 T2(x1') T1(x2')
        # y2 =  5.1 T0(x1') T0(x2') - 0.55 T1(x1') T0(x2') + 0.13 T0(x1') T1(x2') - 0.002 T1(x1') T2(x2')
        coeff_f = np.array([
            [-2.0, 1, 0, 0],
            [0.11, 1, 1, 0],
            [-0.2, 1, 0, 1],
            [0.001, 1, 2, 1],
            [5.1, 2, 0, 0],
            [-0.55, 2, 1, 0],
            [0.13, 2, 0, 1],
            [-0.002, 2, 1, 2]
        ])
        self.assertEqual(nin, coeff_f.shape[1] - 2)

        def referenceFunc(point):
            """Reference implementation; point must be in range [-1, 1]
            """
            c1 = np.zeros((3, 3))
            c1[0, 0] = -2
            c1[1, 0] = 0.11
            c1[0, 1] = -0.2
            c1[2, 1] = 0.001
            c2 = np.zeros((3, 3))
            c2[0, 0] = 5.1
            c2[1, 0] = -0.55
            c2[0, 1] = 0.13
            c2[1, 2] = -0.002
            x1, x2 = point
            return (
                chebval2d(x1, x2, c1),
                chebval2d(x1, x2, c2),
            )

        chebyMap1 = ast.ChebyMap(coeff_f, nout, lbnd_f, ubnd_f)
        self.checkBasicSimplify(chebyMap1)
        self.assertTrue(chebyMap1.hasForward)
        self.assertFalse(chebyMap1.hasInverse)

        outdata = chebyMap1.applyForward(indata)

        referenceCheby = ReferenceCheby(referenceFunc, lbnd_f, ubnd_f)
        des_outdata = referenceCheby.transform(indata)

        npt.assert_allclose(outdata, des_outdata)

        # fit an inverse transform
        chebyMap2 = chebyMap1.polyTran(forward=False, acc=0.0001, maxacc=0.001, maxorder=6,
                                       lbnd=lbnd_f, ubnd=ubnd_f)
        self.assertTrue(chebyMap2.hasForward)
        self.assertTrue(chebyMap2.hasInverse)
        # forward should be identical to the original
        npt.assert_equal(chebyMap2.applyForward(indata), outdata)
        roundTripIn2 = chebyMap2.applyInverse(outdata)
        npt.assert_allclose(roundTripIn2, indata, atol=0.0002)

        # fit an inverse transform with default bounds (which are the same bounds
        # used for fitting chebyMap2, so the results should be the same)
        chebyMap3 = chebyMap1.polyTran(forward=False, acc=0.0001, maxacc=0.001, maxorder=6)
        self.assertTrue(chebyMap2.hasForward)
        self.assertTrue(chebyMap2.hasInverse)
        # forward should be identical to the original
        npt.assert_equal(chebyMap3.applyForward(indata), outdata)
        # inverse should be basically the same
        roundTripIn3 = chebyMap3.applyInverse(outdata)
        npt.assert_allclose(roundTripIn3, roundTripIn2)

    def test_ChebyMapChebyMapUnivertible(self):
        """Test polyTran on a ChebyMap without a single-valued inverse
        """
        nin = 2
        nout = 2
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, 2.5]

        # arbitrary points that cover the input range
        indata = np.array([
            [-2.0, -1.0, 0.1, 1.5, 1.0],
            [0.0, -2.5, -0.2, 2.5, 2.5],
        ])

        # Coefficients for the following not-gently-varying polynomial:
        # y1 = 2.0 T2(x1') T0(x2') - 2.0 T0(x1') T2(x2')
        # y2 =  1.0 T3(x1') T0(x2') - 2.0 T0(x1') T3(x2')
        coeff_f = np.array([
            [2.0, 1, 2, 0],
            [-2.0, 1, 0, 2],
            [1.0, 2, 3, 0],
            [-2.0, 2, 0, 3],
        ])
        self.assertEqual(nin, coeff_f.shape[1] - 2)

        def referenceFunc(point):
            """Reference implementation; point must be in range [-1, 1]
            """
            c1 = np.zeros((3, 3))
            c1[2, 0] = 2.0
            c1[0, 2] = -2.0
            c2 = np.zeros((4, 4))
            c2[3, 0] = 1.0
            c2[0, 3] = -2.0
            x1, x2 = point
            return (
                chebval2d(x1, x2, c1),
                chebval2d(x1, x2, c2),
            )

        chebyMap1 = ast.ChebyMap(coeff_f, nout, lbnd_f, ubnd_f)
        self.checkBasicSimplify(chebyMap1)
        self.assertTrue(chebyMap1.hasForward)
        self.assertFalse(chebyMap1.hasInverse)

        outdata = chebyMap1.applyForward(indata)

        referenceCheby = ReferenceCheby(referenceFunc, lbnd_f, ubnd_f)
        des_outdata = referenceCheby.transform(indata)

        npt.assert_allclose(outdata, des_outdata)

        with self.assertRaises(RuntimeError):
            chebyMap1.polyTran(forward=False, acc=0.0001, maxacc=0.001, maxorder=6,
                               lbnd=lbnd_f, ubnd=ubnd_f)

    def test_chebyGetDomain(self):
        """Test ChebyMap.getDomain's ability to estimate values

        This occurs when there is only one map and you want the inverse
        """
        nout = 2
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, 2.5]

        # Coefficients for the following not-gently-varying polynomial:
        # y1 = 2.0 T2(x1') T0(x2') - 2.0 T0(x1') T2(x2')
        # y2 =  1.0 T3(x1') T0(x2') - 2.0 T0(x1') T3(x2')
        coeff_f = np.array([
            [2.0, 1, 2, 0],
            [-2.0, 1, 0, 2],
            [1.0, 2, 3, 0],
            [-2.0, 2, 0, 3],
        ])

        chebyMap1 = ast.ChebyMap(coeff_f, nout, lbnd_f, ubnd_f)

        # compute indata as a grid of points that cover the input range
        x1Edge = np.linspace(lbnd_f[0], ubnd_f[0], 1000)
        x2Edge = np.linspace(lbnd_f[1], ubnd_f[1], 1000)
        x1Grid, x2Grid = np.meshgrid(x1Edge, x2Edge)
        indata = np.array([x1Grid.ravel(), x2Grid.ravel()])

        outdata = chebyMap1.applyForward(indata)
        pred_lbnd = outdata.min(1)
        pred_ubnd = outdata.max(1)

        domain = chebyMap1.getDomain(forward=False)
        npt.assert_allclose(domain.lbnd, pred_lbnd, atol=0.0001)
        npt.assert_allclose(domain.ubnd, pred_ubnd, atol=0.0001)

    def test_normalize(self):
        """Test the local utility function `normalize`
        """
        lbnd = [-2.0, -2.5]
        ubnd = [1.5, 2.5]

        # points that cover the full domain
        points = np.array([
            [-2.0, -0.5, 0.5, 1.5],
            [-2.5, 1.5, 0.5, 2.5]
        ])

        normPoints = normalize(points, lbnd, ubnd)
        for normAxis in normPoints:
            self.assertAlmostEqual(normAxis.min(), -1)
            self.assertAlmostEqual(normAxis.max(), 1)

    def test_ChebyMapDM10496(self):
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
        lbnd_f = [-2.0, -2.5]
        ubnd_f = [1.5, -0.5]
        lbnd_i = [-3.0]
        ubnd_i = [-1.0]

        # execute many times to increase the odds of a segfault
        for i in range(1000):
            amap = ast.ChebyMap(coeff_f, coeff_i, lbnd_f, ubnd_f, lbnd_i, ubnd_i)
            amapinv = amap.inverted()
            cmp2 = amapinv.then(amap)
            result = cmp2.simplify()
            self.assertIsInstance(result, ast.UnitMap)


if __name__ == "__main__":
    unittest.main()
