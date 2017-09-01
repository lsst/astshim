from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
import numpy.testing as npt

import astshim as ast
from astshim.test import MappingTestCase


class TestUtils(MappingTestCase):

    def test_makePolynomialCoeffs11(self):
        # f(x) = 1.1 + 2.2 x + 3.3 x^2 + 4.4 x^3
        coeffs = [1.1, 2.2, 3.3, 4.4]
        desCoeffArr = np.array([
            [1.1, 1, 0],
            [2.2, 1, 1],
            [3.3, 1, 2],
            [4.4, 1, 3],
        ])

        coeffArr = ast.makePolynomialCoeffs11(coeffs)
        npt.assert_equal(coeffArr, desCoeffArr)

        polyMap = ast.PolyMap(coeffArr, 1)

        indata = np.array([
            [-1.0, 0.0, 1.0, 2.0, 3.0],
        ])
        pred_outdata = [np.polynomial.polynomial.polyval(indata[0], coeffs)]
        outdata = polyMap.applyForward(indata)
        npt.assert_allclose(pred_outdata, outdata)

    def test_makePolynomialCoeffs22(self):
        # f(x,y)[0] = 2.00 + 1.01 y + 1.10 x + 0.02 y^2 + 0.11 x y + 0.20 x^2
        # f(x,y)[1] = 5.00 + 4.01 y + 4.10 x + 3.02 y^2 + 3.11 x y + 3.20 x^2
        xCoeffs = [2.00, 1.01, 1.10, 0.02, 0.11, 0.20]
        yCoeffs = [5.00, 4.01, 4.10, 3.02, 3.11, 3.20]
        desCoeffArr = np.array([
            [2.00, 1, 0, 0],
            [5.00, 2, 0, 0],
            [1.01, 1, 0, 0],
            [4.01, 2, 0, 0],
            [1.10, 1, 1, 0],
            [4.10, 2, 1, 0],
            [0.02, 1, 0, 2],
            [3.02, 2, 0, 2],
            [0.11, 1, 1, 1],
            [3.11, 2, 1, 1],
            [0.20, 1, 2, 0],
            [3.20, 2, 2, 0],
        ])

        coeffArr = ast.makePolynomialCoeffs22(xCoeffs, yCoeffs)
        npt.assert_equal(coeffArr, desCoeffArr)

        polyMap = ast.PolyMap(coeffArr, 1)

        indata = np.array([
            [-1.0, 0.0, 1.0, 2.0, 3.0],
        ])
        pred_outdata = [
            np.polynomial.polynomial.polyval2(indata[0], xCoeffs),
            np.polynomial.polynomial.polyval2(indata[1], yCoeffs),
        ]
        outdata = polyMap.applyForward(indata)
        npt.assert_allclose(pred_outdata, outdata)


if __name__ == "__main__":
    unittest.main()
