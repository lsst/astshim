from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestPcdMap(MappingTestCase):

    def test_PcdMap(self):
        coeff = 0.2
        ctr = [2, 3]
        pcdmap = astshim.PcdMap(coeff, ctr)
        self.assertEqual(pcdmap.getClass(), "PcdMap")
        self.assertIsInstance(pcdmap, astshim.PcdMap)
        self.assertIsInstance(pcdmap, astshim.Mapping)
        self.assertEqual(pcdmap.getNin(), 2)
        self.assertEqual(pcdmap.getNout(), 2)
        self.assertEqual(pcdmap.getDisco(), coeff)
        self.assertAlmostEqual(pcdmap.getPcdCen(1), ctr[0])
        self.assertAlmostEqual(pcdmap.getPcdCen(2), ctr[1])
        self.assertTrue(np.allclose(pcdmap.getPcdCen(), ctr))

        self.checkBasicSimplify(pcdmap)
        self.checkCopy(pcdmap)
        self.checkPersistence(pcdmap)

        # the center maps to itself
        ctrpt = np.array([ctr], dtype=float)
        self.assertTrue(np.allclose(pcdmap.tran(ctrpt), ctrpt))
        return

        indata = np.array([
            [0, 0],
            [-1, 7.3],
            [43.2, -35.3],
        ])
        self.checkRoundTrip(pcdmap, indata)

        outdata = pcdmap.tran(indata)

        # the mapping is:
        #   outrad = inrad*(1 + coeff*inrad^2)
        #   outdir = indir
        # where radius and direction are relative to the center of distortion
        inrelctr = indata - ctr
        inrelctrrad = np.hypot(inrelctr)
        inrelctrdir = np.arctan2(inrelctr[:, 1], inrelctr[:, 0])
        predoutrad = inrelctrrad*(1 + coeff*inrelctrrad*inrelctrrad)
        predoutrelctr = np.zeros(indata.shape, dtype=float)
        predoutrelctr[:, 0] = predoutrad * np.cos(inrelctrdir)
        predoutrelctr[:, 1] = predoutrad * np.sin(inrelctrdir)
        predout = predoutrelctr + ctr
        self.assertTrue(np.allclose(outdata, predout))

    def test_PcdMapBadConstruction(self):
        with self.assertRaises(Exception):
            astshim.PcdMap(0.5, [1, 2, 3])

        with self.assertRaises(Exception):
            astshim.PcdMap(0.5, [1])


if __name__ == "__main__":
    unittest.main()
