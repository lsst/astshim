from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestSlaMap(MappingTestCase):

    def test_SlaMap(self):
        last = 0.1  # an arbitrary value small enough to avoid wrap
        slamap = astshim.SlaMap()
        slamap.add("R2H", [last])
        self.assertEqual(slamap.getClass(), "SlaMap")
        self.assertEqual(slamap.getNin(), 2)
        self.assertEqual(slamap.getNout(), 2)

        self.checkBasicSimplify(slamap)
        self.checkCopy(slamap)
        self.checkPersistence(slamap)

        pin = np.array([
            [0.0, -0.5],
            [1.0, 0.9],
            [3.0, 0.1],
        ])
        pout = slamap.tranForward(pin)
        predpout = pin
        predpout[:, 0] = last - pin[:, 0]
        assert_allclose(pout, predpout)

        self.checkRoundTrip(slamap, pin)


if __name__ == "__main__":
    unittest.main()
