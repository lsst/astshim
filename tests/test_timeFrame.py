from __future__ import absolute_import, division, print_function
import unittest

import astshim as ast
from astshim.test import MappingTestCase


class TestTimeFrame(MappingTestCase):

    def test_TimeFrameBasics(self):
        frame = ast.TimeFrame()
        self.assertEqual(frame.className, "TimeFrame")
        self.assertEqual(frame.nIn, 1)
        self.assertEqual(frame.nAxes, 1)
        self.assertEqual(frame.maxAxes, 1)
        self.assertEqual(frame.minAxes, 1)
        self.assertEqual(frame.alignSystem, "MJD")
        self.assertEqual(frame.dut1, 0.0)
        self.assertEqual(frame.epoch, 2000.0)
        self.assertEqual(frame.obsAlt, 0.0)
        self.assertEqual(frame.obsLat, "N0:00:00.00")
        self.assertEqual(frame.obsLon, "E0:00:00.00")
        self.assertTrue(frame.permute)
        self.assertFalse(frame.preserveAxes)
        self.assertEqual(frame.system, "MJD")
        self.assertEqual(frame.title, "Modified Julian Date")

        self.assertGreater(abs(frame.getBottom(1)), 1e99)
        self.assertGreater(abs(frame.getTop(1)), 1e99)
        self.assertGreater(frame.getTop(1), frame.getBottom(1))
        self.assertTrue(frame.getDirection(1))
        self.assertEqual(frame.getInternalUnit(1), "d")
        self.assertEqual(frame.getNormUnit(1), "")
        self.assertEqual(frame.getSymbol(1), "MJD")
        self.assertEqual(frame.getUnit(1), "d")

        self.assertEqual(frame.alignTimeScale, "TAI")
        self.assertEqual(frame.ltOffset, 0.0)
        self.assertEqual(frame.timeOrigin, 0.0)
        self.assertEqual(frame.timeScale, "TAI")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def testTimeFrameAttributes(self):
        frame = ast.TimeFrame(
            "AlignTimeScale=TT, LTOffset=1.1, TimeOrigin=2.2, TimeScale=TDB")

        self.assertEqual(frame.alignTimeScale, "TT")
        self.assertAlmostEqual(frame.ltOffset, 1.1, places=3)
        self.assertAlmostEqual(frame.timeOrigin, 2.2, places=3)
        self.assertEqual(frame.timeScale, "TDB")

        frame.alignTimeScale = "LMST"
        frame.ltOffset = 55.5
        frame.timeOrigin = 66.6
        frame.timeScale = "UT1"
        self.assertEqual(frame.alignTimeScale, "LMST")
        self.assertAlmostEqual(frame.ltOffset, 55.5, places=3)
        self.assertAlmostEqual(frame.timeOrigin, 66.6, places=3)
        self.assertEqual(frame.timeScale, "UT1")


if __name__ == "__main__":
    unittest.main()
