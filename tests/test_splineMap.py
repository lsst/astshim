import unittest

import numpy as np
import numpy.testing as npt
from scipy.interpolate import RectBivariateSpline

import astshim as ast
from astshim.test import MappingTestCase


class TestSplineMap(MappingTestCase):

    def setUp(self):
        self.nx = 150
        self.ny = 100
        self.k = 4
        a = 4.0
        b = 0.2
        c = 3.0

        self.xs = np.linspace(1, self.nx, self.nx)
        self.ys = np.linspace(1, self.ny, self.ny)

        fcx = np.zeros((self.nx, self.ny))
        fcy = np.zeros((self.nx, self.ny))
        for j in range(self.ny):
            y = self.ys[j]
            for i in range(self.nx):
                x = self.xs[i]
                fcx[i][j] = x + a * np.sin(b * x) * np.cos(b * y)
                fcy[i][j] = y + c * np.cos(b * x) * np.sin(b * y)

        self.splinex = RectBivariateSpline(self.xs, self.ys, fcx, s=0)
        (self.tx, self.ty) = self.splinex.get_knots()
        self.arx = self.splinex.get_coeffs()

        self.spliney = RectBivariateSpline(self.xs, self.ys, fcy, s=0)
        self.ary = self.spliney.get_coeffs()

        self.splineMap = ast.SplineMap(
            self.k, self.k, self.nx, self.ny, self.tx, self.ty, self.arx, self.ary
        )

    def test_SplineMap(self):
        """Test that the forward and inverse transforms match
        scipy.interpolate.RectBivariateSpline.
        """
        xval = np.array([0.0, 12.5, 12.0, 1.0, 15.0, 1.0, 95.0, 149.5, 151.2, 77.77])
        yval = np.array([-1.0, 8.8, 8.0, 1.0, 15.0, 76.0, 100.0, 99.8, 82.3, 54.3])

        u = self.splinex.ev(xval, yval)
        v = self.spliney.ev(xval, yval)

        outData = self.splineMap.applyForward(np.array([xval, yval]))

        outOfBounds = (
            (xval > self.xs.max())
            | (xval < self.xs.min())
            | (yval < self.ys.min())
            | (yval > self.ys.max())
        )

        npt.assert_equal(outData[:, outOfBounds], np.nan)
        npt.assert_almost_equal(outData[0, ~outOfBounds], u[~outOfBounds])
        npt.assert_almost_equal(outData[1, ~outOfBounds], v[~outOfBounds])

        reverseTrip = self.splineMap.applyInverse(outData[:, ~outOfBounds].copy())

        npt.assert_almost_equal(reverseTrip[0], xval[~outOfBounds])
        npt.assert_almost_equal(reverseTrip[1], yval[~outOfBounds])

    def test_splineMapAttributes(self):
        """Check that SplineMap attributes can be accessed and return expected
        values."""
        self.assertEqual(self.splineMap.invTol, 1e-6)
        self.assertEqual(self.splineMap.invNIter, 6)
        self.assertEqual(self.splineMap.outUnit, False)

        # Initialize SplineMap with other attributes and check they are set
        # correctly.
        options = "InvNIter=5,InvTol=1e-7,OutUnit=1"
        splineMap = ast.SplineMap(
            self.k,
            self.k,
            self.nx,
            self.ny,
            self.tx,
            self.ty,
            self.arx,
            self.ary,
            options=options,
        )
        self.assertEqual(splineMap.invTol, 1e-7)
        self.assertEqual(splineMap.invNIter, 5)
        self.assertEqual(splineMap.outUnit, 1)


if __name__ == "__main__":
    unittest.main()
