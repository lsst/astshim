from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestMapBox(MappingTestCase):

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
        # If maxOutCoord is not specified by the user, it should be set to nout
        self.assertEqual(mapbox.maxOutCoord, winmap.nOut)

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


if __name__ == "__main__":
    unittest.main()
