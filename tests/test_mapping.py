from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestMapping(MappingTestCase):
    """Test basics of Mapping

    Note that Mapping.of and Mapping.over are tested by test_cmpMap.py
    """

    def setUp(self):
        self.nin = 2
        self.zoom = 1.3
        self.zoommap = astshim.ZoomMap(self.nin, self.zoom)

    def test_MappingAttributes(self):
        self.assertEqual(self.zoommap.getClassName(), "ZoomMap")
        self.assertFalse(self.zoommap.isInverted())
        self.assertTrue(self.zoommap.getIsLinear())
        self.assertFalse(self.zoommap.getIsSimple())
        self.assertEqual(self.zoommap.getNIn(), self.nin)
        self.assertEqual(self.zoommap.getNOut(), self.nin)
        self.assertFalse(self.zoommap.getReport())
        self.assertTrue(self.zoommap.hasForward())
        self.assertTrue(self.zoommap.hasInverse())

    def test_MappingInvert(self):
        invmap = self.zoommap.getInverse()

        self.assertEqual(invmap.getClassName(), "ZoomMap")
        self.assertTrue(invmap.isInverted())
        self.assertTrue(invmap.getIsLinear())
        self.assertFalse(invmap.getIsSimple())
        self.assertTrue(invmap.hasForward())
        self.assertTrue(invmap.hasInverse())

        indata = np.array([
            [1.0, 2.0, -6.0, 30.0, 0.0],
            [3.0, 99.0, -5.0, 21.0, 0.0],
        ], dtype=float)
        self.checkRoundTrip(self.zoommap, indata)
        self.checkRoundTrip(invmap, indata)

    def test_MapBox(self):
        """Test MapBox for the simple case of a shift and zoom"""
        shift = np.array([1.5, 0.5])
        zoom = np.array([2.0, 3.0])
        winmap = astshim.WinMap(
            [0, 0], [1, 1], zoom * [0, 0] + shift, zoom * [1, 1] + shift)
        # arbitrary values chosen so that inbnd_a is NOT < inbnd_b for both axes because
        # MapBox uses the minimum of inbnd_b, inbnd_a for each axis for the lower bound,
        # and the maximum for the upper bound
        inbnd_a = np.array([-1.2, 3.3])
        inbnd_b = np.array([2.7, 2.2])
        mapbox = astshim.MapBox(winmap, inbnd_a, inbnd_b)

        lbndin = np.minimum(inbnd_a, inbnd_b)
        ubndin = np.maximum(inbnd_a, inbnd_b)
        predlbndOut = lbndin * zoom + shift
        predubndOut = ubndin * zoom + shift
        assert_allclose(mapbox.lbndOut, predlbndOut)
        assert_allclose(mapbox.ubndOut, predubndOut)

        # note that mapbox.xl and xu is only partially predictable
        # because any X from the input gives the same Y
        for i in range(2):
            self.assertAlmostEqual(mapbox.xl[i, i], lbndin[i])
            self.assertAlmostEqual(mapbox.xu[i, i], ubndin[i])

        # confirm that order of inbnd_a, inbnd_b doesn't matter
        mapbox2 = astshim.MapBox(winmap, inbnd_b, inbnd_a)
        assert_allclose(mapbox2.lbndOut, mapbox.lbndOut)
        assert_allclose(mapbox2.ubndOut, mapbox.ubndOut)

        # the xl and xu need only agree on the diagonal, as above
        for i in range(2):
            self.assertAlmostEqual(mapbox.xl[i, i], mapbox2.xl[i, i])
            self.assertAlmostEqual(mapbox.xu[i, i], mapbox2.xu[i, i])

    def test_MappingLinearApprox(self):
        """Exercise Mapping.linearApprox for a trivial case"""
        coeffs = self.zoommap.linearApprox([0, 0], [50, 50], 1e-5)
        descoeffs = np.array([
            [0.0, 0.0],
            [1.3, 0.0],
            [0.0, 1.3]
        ], dtype=float)
        assert_allclose(coeffs, descoeffs)

    def test_QuadApprox(self):
        # simple parabola
        coeff_f = np.array([
            [0.5, 1, 2, 0],
            [0.5, 1, 0, 2],
        ], dtype=float)
        polymap = astshim.PolyMap(coeff_f, 1)
        qa = astshim.QuadApprox(polymap, [-1, -1], [1, 1], 3, 3)
        self.assertAlmostEqual(qa.rms, 0)
        self.assertEqual(len(qa.fit), 6)
        assert_allclose(qa.fit, [0, 0, 0, 0, 0.5, 0.5])

    def test_MappingRate(self):
        """Exercise Mapping.rate for a trivial case"""
        for x in (0, 5, 55):  # arbitrary, but include 0
            for y in (0, -9.5, 47.6):   # arbitrary, but include 0
                for xaxis in (1, 2):
                    for yaxis in (1, 2):
                        desrate = self.zoom if xaxis == yaxis else 0
                        self.assertAlmostEqual(self.zoommap.rate(
                            [x, y], xaxis, yaxis), desrate)

    def test_MappingSetReport(self):
        self.assertFalse(self.zoommap.getReport())
        self.assertFalse(self.zoommap.test("Report"))
        self.zoommap.setReport(False)
        self.assertFalse(self.zoommap.getReport())
        self.assertTrue(self.zoommap.test("Report"))
        self.zoommap.setReport(True)
        self.assertTrue(self.zoommap.getReport())
        self.assertTrue(self.zoommap.test("Report"))
        self.zoommap.clear("Report")
        self.assertFalse(self.zoommap.getReport())
        self.assertFalse(self.zoommap.test("Report"))

    def test_MappingSimplify(self):
        simpmap = self.zoommap.simplify()

        self.assertEqual(simpmap.getClassName(), "ZoomMap")
        self.assertFalse(simpmap.isInverted())
        self.assertTrue(simpmap.getIsSimple())
        self.assertEqual(simpmap.getNIn(), self.nin)
        self.assertEqual(simpmap.getNOut(), self.nin)
        self.assertTrue(simpmap.hasForward())
        self.assertTrue(simpmap.hasInverse())

    def test_MapSplit(self):
        """Test MapSplit for a simple case"""
        for i in range(self.nin):
            split = astshim.MapSplit(self.zoommap, [i + 1])
            self.assertEqual(split.splitMap.getClassName(), "ZoomMap")
            self.assertEqual(split.splitMap.getNIn(), 1)
            self.assertEqual(split.splitMap.getNOut(), 1)
            self.assertEqual(split.origOut[0], i + 1)


if __name__ == "__main__":
    unittest.main()
