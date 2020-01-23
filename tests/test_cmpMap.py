import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestCmpMap(MappingTestCase):

    """Test compound maps: CmpMap, ParallelMap and SeriesMap
    """

    def setUp(self):
        self.nin = 2
        self.zoom = 1.3
        self.shift = [-0.5, 1.2]
        self.zoommap = ast.ZoomMap(self.nin, self.zoom)
        self.shiftmap = ast.ShiftMap(self.shift)

    def test_SeriesMap(self):
        sermap = ast.SeriesMap(self.shiftmap, self.zoommap)
        self.assertEqual(sermap.getRefCount(), 1)

        self.checkBasicSimplify(sermap)
        self.checkCopy(sermap)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, sermap, isSeries=True)

        sermap2 = self.shiftmap.then(self.zoommap)
        self.checkBasicSimplify(sermap2)
        self.checkCopy(sermap2)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, sermap2, isSeries=True)

        sermap3 = ast.CmpMap(self.shiftmap, self.zoommap, True)
        self.checkBasicSimplify(sermap3)
        self.checkCopy(sermap3)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, sermap3, isSeries=True)

        indata = np.array([
            [1.0, 2.0, -6.0, 30.0, 0.2],
            [3.0, 99.9, -5.1, 21.0, 0.0],
        ], dtype=float)
        pred_outdata = ((indata.T + self.shift) * self.zoom).T
        topos = sermap.applyForward(indata)
        assert_allclose(topos, pred_outdata)

        topos2 = sermap2.applyForward(indata)
        assert_allclose(topos2, pred_outdata)

        topos3 = sermap3.applyForward(indata)
        assert_allclose(topos3, pred_outdata)

        self.checkRoundTrip(sermap, indata)
        self.checkRoundTrip(sermap2, indata)
        self.checkRoundTrip(sermap3, indata)

        self.checkMappingPersistence(sermap, indata)
        self.checkMappingPersistence(sermap2, indata)
        self.checkMappingPersistence(sermap3, indata)

    def test_ParallelMap(self):
        parmap = ast.ParallelMap(self.shiftmap, self.zoommap)
        # adding to a ParallelMap increases by 1
        self.assertEqual(self.shiftmap.getRefCount(), 2)
        # adding to a ParallelMap increases by 1
        self.assertEqual(self.zoommap.getRefCount(), 2)
        self.assertEqual(parmap.nIn, self.nin * 2)
        self.assertEqual(parmap.nOut, self.nin * 2)
        self.assertFalse(parmap.series)

        self.checkBasicSimplify(parmap)
        self.checkCopy(parmap)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, parmap, isSeries=False)

        parmap2 = self.shiftmap.under(self.zoommap)
        self.checkBasicSimplify(parmap2)
        self.checkCopy(parmap2)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, parmap2, isSeries=False)

        indata = np.array([
            [3.0, 1.0, -6.0],
            [2.2, 3.0, -5.1],
            [-5.6, 2.0, 30.0],
            [0.32, 99.9, 21.0],
        ], dtype=float)
        pred_outdata = indata.copy()
        pred_outdata.T[:, 0:2] += self.shift
        pred_outdata.T[:, 2:4] *= self.zoom
        topos = parmap.applyForward(indata)
        assert_allclose(topos, pred_outdata)

        topos2 = parmap2.applyForward(indata)
        assert_allclose(topos2, pred_outdata)

        parmap3 = ast.CmpMap(self.shiftmap, self.zoommap, False)
        self.checkBasicSimplify(parmap3)
        self.checkCopy(parmap3)
        self.checkMemoryForCompoundObject(self.shiftmap, self.zoommap, parmap3, isSeries=False)

        topos3 = parmap3.applyForward(indata)
        assert_allclose(topos3, pred_outdata)

        self.checkRoundTrip(parmap, indata)
        self.checkRoundTrip(parmap2, indata)
        self.checkRoundTrip(parmap3, indata)

        self.checkMappingPersistence(parmap, indata)
        self.checkMappingPersistence(parmap2, indata)
        self.checkMappingPersistence(parmap3, indata)

    def test_SeriesMapMatrixShiftSimplify(self):
        """Test that a non-square matrix map followed by a shift map can be
        simplified.

        This is ticket DM-10946
        """
        m1 = 1.0
        m2 = 2.0
        shift = 3.0
        matrixMap = ast.MatrixMap(np.array([[m1, m2]]))
        self.assertEqual(matrixMap.nIn, 2)
        self.assertEqual(matrixMap.nOut, 1)
        shiftMap = ast.ShiftMap([shift])
        seriesMap = matrixMap.then(shiftMap)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ], dtype=float)
        pred_outdata = m1 * indata[0] + m2 * indata[1] + shift
        pred_outdata.shape = (1, len(pred_outdata))

        outdata = seriesMap.applyForward(indata)
        assert_allclose(outdata, pred_outdata)

        simplifiedMap = seriesMap.simplified()
        outdata2 = simplifiedMap.applyForward(indata)
        assert_allclose(outdata2, pred_outdata)


if __name__ == "__main__":
    unittest.main()
