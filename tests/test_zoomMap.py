from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestZoomMap(MappingTestCase):

    def test_basics(self):
        """Test basics of ZoomMap including applyForward
        """
        for nin in (1, 2, 3):
            for zoom in (1.0, -1.1, 359.3):
                zoommap = ast.ZoomMap(nin, zoom)
                self.assertEqual(zoommap.className, "ZoomMap")
                self.assertEqual(zoommap.nIn, nin)
                self.assertEqual(zoommap.nOut, nin)
                self.assertTrue(zoommap.isLinear)

                self.checkBasicSimplify(zoommap)
                self.checkCopy(zoommap)

                indata = np.array([
                    [1.0, 2.0, -6.0, 30.0, 1.0],
                    [3.0, 99.0, -5.0, 21.0, 0.0],
                    [-5.0, 3.0, -7.0, 37.0, 0.0],
                    [7.0, -23.0, -3.0, 45.0, 0.0],
                ], dtype=float)[0:nin]
                self.checkRoundTrip(zoommap, indata)
                self.checkMappingPersistence(zoommap, indata)

                topos = zoommap.applyForward(indata)
                assert_allclose(indata * zoom, topos)


if __name__ == "__main__":
    unittest.main()
