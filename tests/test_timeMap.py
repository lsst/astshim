from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase

SecPerDay = 3600*24


class TestTimeMap(MappingTestCase):

    def test_TimeMapDefault(self):
        """Test a TimeMap with no conversions added
        """
        timemap = astshim.TimeMap()
        self.assertEqual(timemap.getClass(), "TimeMap")
        self.assertEqual(timemap.getNin(), 1)
        self.assertEqual(timemap.getNout(), 1)

        self.checkCopy(timemap)
        self.checkPersistence(timemap)
        self.checkBasicSimplify(timemap)

        indata = np.array([
            [0],
            [1],
        ], dtype=float)
        outdata = timemap.tran(indata)
        self.assertTrue(np.allclose(outdata, indata))

        self.checkRoundTrip(timemap, indata)

    def test_TimeMapAddUTTOUTC(self):
        timemap = astshim.TimeMap()
        dut1 = 0.35
        timemap.add("UTTOUTC", [dut1])
        indata = np.array([
            [512345],
            [512346],
        ], dtype=float)
        outdata = timemap.tran(indata)
        predoutdata = indata - dut1/SecPerDay
        self.assertTrue(np.allclose(outdata, predoutdata, atol=1e-15, rtol=0))

        self.checkRoundTrip(timemap, indata)

    def test_TimeMapAddInvalid(self):
        timemap = astshim.TimeMap()

        with self.assertRaises(Exception):
            timemap.add("BEPTOMJD", [560])  # too few arguments
        with self.assertRaises(Exception):
            timemap.add("BEPTOMJD", [560, 720, 33])  # too many arguments
        timemap.add("BEPTOMJD", [560, 720])  # just right

        with self.assertRaises(Exception):
            timemap.add("UTCTOTAI", [])  # too few arguments
        with self.assertRaises(Exception):
            timemap.add("UTCTOTAI", [560, 720])  # too many arguments
        timemap.add("UTCTOTAI", [560])  # just right

        with self.assertRaises(Exception):
            timemap.add("TTTOTDB", [560, 720, 33])  # too few arguments
        with self.assertRaises(Exception):
            timemap.add("TTTOTDB", [560, 720, 33, 53, 23])  # too many arguments
        timemap.add("TTTOTDB", [560, 720, 33, 53])  # just right

        with self.assertRaises(Exception):
            timemap.timeadd("UNRECOGNIZED", [1])


if __name__ == "__main__":
    unittest.main()
