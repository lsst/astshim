from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestCmpMap(MappingTestCase):

    """Test compound maps: CmpMap, ParallelMap and SeriesMap
    """

    def setUp(self):
        self.nin = 2
        self.zoom = 1.3
        self.shift = [-0.5, 1.2]
        self.zoommap = astshim.ZoomMap(self.nin, self.zoom)
        self.shiftmap = astshim.ShiftMap(self.shift)

    def test_SeriesMap(self):
        sermap = astshim.SeriesMap(self.shiftmap, self.zoommap)
        # adding to a SeriesMap increases by 1
        self.assertEqual(self.shiftmap.getRefCount(), 2)
        # adding to a SeriesMap increases by 1
        self.assertEqual(self.zoommap.getRefCount(), 2)
        self.assertEqual(sermap.getNin(), self.nin)
        self.assertEqual(sermap.getNout(), self.nin)
        self.assertTrue(sermap.getSeries())

        self.checkBasicSimplify(sermap)
        self.checkCopy(sermap)
        self.checkPersistence(sermap)

        indata = np.array([
            [1, 3],
            [2, 99.9],
            [-6, -5.1],
            [30, 21],
            [0.2, 0],
        ], dtype=float)
        pred_outdata = (indata + self.shift) * self.zoom
        topos = sermap.tranForward(indata)
        assert_allclose(topos, pred_outdata)

        self.checkRoundTrip(sermap, indata)

        cmpmap = astshim.CmpMap(self.shiftmap, self.zoommap, True)
        cmtopos = cmpmap.tranForward(indata)
        assert_allclose(cmtopos, pred_outdata)

    def test_ParallelMap(self):
        parmap = astshim.ParallelMap(self.shiftmap, self.zoommap)
        # adding to a ParallelMap increases by 1
        self.assertEqual(self.shiftmap.getRefCount(), 2)
        # adding to a ParallelMap increases by 1
        self.assertEqual(self.zoommap.getRefCount(), 2)
        self.assertEqual(parmap.getNin(), self.nin * 2)
        self.assertEqual(parmap.getNout(), self.nin * 2)
        self.assertFalse(parmap.getSeries())

        self.checkBasicSimplify(parmap)
        self.checkCopy(parmap)
        self.checkPersistence(parmap)

        indata = np.array([
            [-3, 2.2, -5.6, 0.32],
            [1, 3, 2, 99.9],
            [-6, -5.1, 30, 21],
        ], dtype=float)
        pred_outdata = indata.copy()
        pred_outdata[:, 0:2] += self.shift
        pred_outdata[:, 2:4] *= self.zoom
        topos = parmap.tranForward(indata)
        assert_allclose(topos, pred_outdata)

        self.checkRoundTrip(parmap, indata)

        cmpmap = astshim.CmpMap(self.shiftmap, self.zoommap, False)
        cmtopos = cmpmap.tranForward(indata)
        assert_allclose(cmtopos, pred_outdata)


if __name__ == "__main__":
    unittest.main()
