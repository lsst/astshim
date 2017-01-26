from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestZoomMap(MappingTestCase):

    def test_basics(self):
        """Test basics of ZoomMap including tran
        """
        for nin in (1, 2, 3):
            for zoom in (1.0, -1.1, 359.3):
                zoommap = astshim.ZoomMap(nin, zoom)
                self.assertEqual(zoommap.getClass(), "ZoomMap")
                self.assertEqual(zoommap.getNin(), nin)
                self.assertEqual(zoommap.getNout(), nin)
                self.assertTrue(zoommap.getIsLinear())

                self.checkBasicSimplify(zoommap)
                self.checkCopy(zoommap)
                self.checkPersistence(zoommap)

                frompos = np.array([
                    [1, 3, -5, 7],
                    [2, 99, 3, -23],
                    [-6, -5, -7, -3],
                    [30, 21, 37, 45],
                    [1, 0, 0, 0],
                ], dtype=float)
                frompos = np.array(frompos[:, 0:nin])
                self.checkRoundTrip(zoommap, frompos)

                topos = zoommap.tran(frompos)
                self.assertTrue(np.allclose(frompos * zoom, topos))


if __name__ == "__main__":
    unittest.main()
