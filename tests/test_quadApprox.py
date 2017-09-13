from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestQuadApprox(MappingTestCase):

    def test_QuadApprox(self):
        # simple parabola
        coeff_f = np.array([
            [0.5, 1, 2, 0],
            [0.5, 1, 0, 2],
        ], dtype=float)
        polymap = ast.PolyMap(coeff_f, 1)
        qa = ast.QuadApprox(polymap, [-1, -1], [1, 1], 3, 3)
        self.assertAlmostEqual(qa.rms, 0)
        self.assertEqual(len(qa.fit), 6)
        assert_allclose(qa.fit, [0, 0, 0, 0, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
