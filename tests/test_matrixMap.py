from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestMatrixMap(MappingTestCase):

    def test_MatrixMapDiagonal(self):
        """Test MatrixMap constructed with a diagonal vector"""

        mm = astshim.MatrixMap([-1.0, 2.0])
        self.assertEqual(mm.getClass(), "MatrixMap")
        self.assertEqual(mm.getNin(), 2)
        self.assertEqual(mm.getNout(), 2)
        self.assertTrue(mm.getTranForward())
        self.assertTrue(mm.getTranInverse())

        self.checkBasicSimplify(mm)
        self.checkCopy(mm)
        self.checkPersistence(mm)

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ], dtype=float)
        pout = mm.tran(pin)
        despout = np.array([
            [-1.0, 0.0],
            [-2.0, 2.0],
            [-3.0, 4.0],
        ], dtype=float)
        self.assertTrue(np.allclose(pout, despout))

        self.checkRoundTrip(mm, pin)

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
        self.assertEqual(mm.getNin(), 2)
        self.assertEqual(mm.getNout(), 3)
        self.assertTrue(mm.getTranForward())
        self.assertFalse(mm.getTranInverse())

        pin = np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ], dtype=float)
        pout = mm.tran(pin)
        despout = np.array([
            [0.0, 2.0, -1.0],
            [1.0, 7.0, -4.0],
            [2.0, 12.0, -7.0],
        ], dtype=float)
        self.assertTrue(np.allclose(pout, despout))

if __name__ == "__main__":
    unittest.main()
