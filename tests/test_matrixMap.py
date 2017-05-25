from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestMatrixMap(MappingTestCase):

    def test_MatrixMapDiagonal(self):
        """Test MatrixMap constructed with a diagonal vector"""

        mm = astshim.MatrixMap([-1.0, 2.0])
        self.assertEqual(mm.className, "MatrixMap")
        self.assertEqual(mm.nIn, 2)
        self.assertEqual(mm.nOut, 2)
        self.assertTrue(mm.hasForward)
        self.assertTrue(mm.hasInverse)

        self.checkBasicSimplify(mm)
        self.checkCopy(mm)
        self.checkPersistence(mm)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ], dtype=float)
        outdata = mm.tranForward(indata)
        pred_outdata = np.array([
            [-1.0, -2.0, -3.0],
            [0.0, 2.0, 4.0],
        ], dtype=float)
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(mm, indata)

    def test_MatrixMapMatrix(self):
        """Test MatrixMap constructed with a 2-d matrix

        matrix    inputs    expected outputs
         0,  1    (1, 0)    (0, 2, -1)
         2,  3    (2, 1)    (1, 7, -4)
        -1, -2    (3, 2)    (2, 12, -7)
        """
        matrix = np.array([
            [0.0, 1.0],
            [2.0, 3.0],
            [-1.0, -2.0]
        ], dtype=float)
        mm = astshim.MatrixMap(matrix)
        self.assertEqual(mm.nIn, 2)
        self.assertEqual(mm.nOut, 3)
        self.assertTrue(mm.hasForward)
        self.assertFalse(mm.hasInverse)

        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ], dtype=float)
        outdata = mm.tranForward(indata)
        pred_outdata = np.array([
            [0.0, 1.0, 2.0],
            [2.0, 7.0, 12.0],
            [-1.0, -4.0, -7.0],
        ], dtype=float)
        assert_allclose(outdata, pred_outdata)


if __name__ == "__main__":
    unittest.main()
