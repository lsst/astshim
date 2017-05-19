from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestZoomMap(MappingTestCase):

    def test_basics(self):
        """Test basics of ZoomMap including tranForward
        """
        for nin in (1, 2, 3):
            for zoom in (1.0, -1.1, 359.3):
                zoommap = astshim.ZoomMap(nin, zoom)
                self.assertEqual(zoommap.getClass(), "ZoomMap")
                self.assertEqual(zoommap.getNIn(), nin)
                self.assertEqual(zoommap.getNOut(), nin)
                self.assertTrue(zoommap.getIsLinear())

                self.checkBasicSimplify(zoommap)
                self.checkCopy(zoommap)
                self.checkPersistence(zoommap)

                indata = np.array([
                    [1.0, 2.0, -6.0, 30.0, 1.0],
                    [3.0, 99.0, -5.0, 21.0, 0.0],
                    [-5.0, 3.0, -7.0, 37.0, 0.0],
                    [7.0, -23.0, -3.0, 45.0, 0.0],
                ], dtype=float)[0:nin]
                self.checkRoundTrip(zoommap, indata)

                topos = zoommap.tranForward(indata)
                assert_allclose(indata * zoom, topos)


if __name__ == "__main__":
    unittest.main()
