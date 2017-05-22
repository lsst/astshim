from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase

SecPerDay = 3600 * 24


class TestTimeMap(MappingTestCase):

    def test_TimeMapDefault(self):
        """Test a TimeMap with no conversions added
        """
        timemap = astshim.TimeMap()
        self.assertEqual(timemap.getClassName(), "TimeMap")
        self.assertEqual(timemap.getNIn(), 1)
        self.assertEqual(timemap.getNOut(), 1)

        self.checkCopy(timemap)
        self.checkPersistence(timemap)
        self.checkBasicSimplify(timemap)

        indata = np.array([0.0, 1.0], dtype=float)
        outdata = timemap.tranForward(indata)
        assert_allclose(outdata, indata)

        self.checkRoundTrip(timemap, indata)

    def test_TimeMapAddUTTOUTC(self):
        timemap = astshim.TimeMap()
        dut1 = 0.35
        timemap.add("UTTOUTC", [dut1])
        indata = np.array([512345.0, 512346.0], dtype=float)
        outdata = timemap.tranForward(indata)
        pred_outdata = (indata.T - dut1 / SecPerDay).T
        assert_allclose(outdata, pred_outdata, atol=1e-15, rtol=0)

        self.checkRoundTrip(timemap, indata)

    def test_TimeMapAddInvalid(self):
        timemap = astshim.TimeMap()

        with self.assertRaises(Exception):
            timemap.add("BEPTOMJD", [560])  # too few arguments
        with self.assertRaises(Exception):
            timemap.add("BEPTOMJD", [560, 720, 33])  # too many arguments
        timemap.add("BEPTOMJD", [560, 720])  # just right

        with self.assertRaises(Exception):
            timemap.add("UTCTOTAI", [560])  # just right
        with self.assertRaises(Exception):
            timemap.add("UTCTOTAI", [560, 720, 23])  # too many arguments
        timemap.add("UTCTOTAI", [560, 720])  # too many arguments

        with self.assertRaises(Exception):
            timemap.add("TTTOTDB", [560, 720, 33, 53])  # too few arguments
        with self.assertRaises(Exception):
            # too many arguments
            timemap.add("TTTOTDB", [560, 720, 33, 53, 23, 10, 20])
        timemap.add("TTTOTDB", [560, 720, 33, 53, 10])  # just right

        with self.assertRaises(Exception):
            timemap.timeadd("UNRECOGNIZED", [1])


if __name__ == "__main__":
    unittest.main()
