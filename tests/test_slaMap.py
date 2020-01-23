import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestSlaMap(MappingTestCase):

    def test_SlaMap(self):
        last = 0.1  # an arbitrary value small enough to avoid wrap
        slamap = ast.SlaMap()
        slamap.add("R2H", [last])
        self.assertEqual(slamap.className, "SlaMap")
        self.assertEqual(slamap.nIn, 2)
        self.assertEqual(slamap.nOut, 2)

        self.checkBasicSimplify(slamap)
        self.checkCopy(slamap)

        indata = np.array([
            [0.0, 1.0, 3.0],
            [-0.5, 0.9, 0.1],
        ])
        outdata = slamap.applyForward(indata)
        pred_outdata = indata
        pred_outdata[0] = last - indata[0]
        assert_allclose(outdata, pred_outdata)

        self.checkRoundTrip(slamap, indata)
        self.checkMappingPersistence(slamap, indata)


if __name__ == "__main__":
    unittest.main()
