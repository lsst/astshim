from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestPcdMap(MappingTestCase):

    def test_PcdMap(self):
        coeff = 0.002
        ctr = [2, 3]
        pcdmap = ast.PcdMap(coeff, ctr)
        self.assertEqual(pcdmap.className, "PcdMap")
        self.assertIsInstance(pcdmap, ast.PcdMap)
        self.assertIsInstance(pcdmap, ast.Mapping)
        self.assertEqual(pcdmap.nIn, 2)
        self.assertEqual(pcdmap.nOut, 2)
        self.assertEqual(pcdmap.disco, coeff)
        assert_allclose(pcdmap.pcdCen, ctr)

        self.checkBasicSimplify(pcdmap)
        self.checkCopy(pcdmap)

        # the center maps to itself
        assert_allclose(pcdmap.applyForward(ctr), ctr)

        indata = np.array([
            [0.0, -1.0, 4.2],
            [0.0, 7.3, -5.3],
        ])
        # inverse uses a fit so don't expect too much
        self.checkRoundTrip(pcdmap, indata, atol=1e-4)
        self.checkMappingPersistence(pcdmap, indata)

        outdata = pcdmap.applyForward(indata)

        # the mapping is:
        #   outrad = inrad*(1 + coeff*inrad^2)
        #   outdir = indir
        # where radius and direction are relative to the center of distortion
        inrelctr = (indata.T - ctr).T
        inrelctrrad = np.hypot(inrelctr[0], inrelctr[1])
        inrelctrdir = np.arctan2(inrelctr[1], inrelctr[0])
        pred_outrad = inrelctrrad * (1 + coeff * inrelctrrad * inrelctrrad)
        pred_outrelctr = np.zeros(indata.shape, dtype=float)
        pred_outrelctr[0, :] = pred_outrad * np.cos(inrelctrdir)
        pred_outrelctr[1, :] = pred_outrad * np.sin(inrelctrdir)
        pred_outdata = pred_outrelctr + np.expand_dims(ctr, 1)
        assert_allclose(outdata, pred_outdata)

    def test_PcdMapBadConstruction(self):
        with self.assertRaises(Exception):
            ast.PcdMap(0.5, [1, 2, 3])

        with self.assertRaises(Exception):
            ast.PcdMap(0.5, [1])


if __name__ == "__main__":
    unittest.main()
