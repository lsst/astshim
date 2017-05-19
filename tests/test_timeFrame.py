from __future__ import absolute_import, division, print_function
import math
import unittest

import astshim
from astshim.test import MappingTestCase


class TestTimeFrame(MappingTestCase):

    def test_TimeFrameBasics(self):
        frame = astshim.TimeFrame()
        self.assertEqual(frame.getClass(), "TimeFrame")
        self.assertEqual(frame.getNIn(), 1)
        self.assertEqual(frame.getNAxes(), 1)
        self.assertEqual(frame.getMaxAxes(), 1)
        self.assertEqual(frame.getMinAxes(), 1)
        self.assertEqual(frame.getAlignSystem(), "MJD")
        self.assertEqual(frame.getDut1(), 0.0)
        self.assertEqual(frame.getEpoch(), 2000.0)
        self.assertEqual(frame.getObsAlt(), 0.0)
        self.assertEqual(frame.getObsLat(), "N0:00:00.00")
        self.assertEqual(frame.getObsLon(), "E0:00:00.00")
        self.assertTrue(frame.getPermute())
        self.assertFalse(frame.getPreserveAxes())
        self.assertEqual(frame.getSystem(), "MJD")
        self.assertEqual(frame.getTitle(), "Modified Julian Date")

        self.assertTrue(math.isinf(frame.getBottom(1)))
        self.assertTrue(math.isinf(frame.getTop(1)))
        self.assertGreater(frame.getTop(1), frame.getBottom(1))
        self.assertTrue(frame.getDirection(1))
        self.assertEqual(frame.getInternalUnit(1), "d")
        self.assertEqual(frame.getNormUnit(1), "")
        self.assertEqual(frame.getSymbol(1), "MJD")
        self.assertEqual(frame.getUnit(1), "d")

        self.assertEqual(frame.getAlignTimeScale(), "TAI")
        self.assertEqual(frame.getLTOffset(), 0.0)
        self.assertEqual(frame.getTimeOrigin(), 0.0)
        self.assertEqual(frame.getTimeScale(), "TAI")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def testTimeFrameAttributes(self):
        frame = astshim.TimeFrame(
            "AlignTimeScale=TT, LTOffset=1.1, TimeOrigin=2.2, TimeScale=TDB")

        self.assertEqual(frame.getAlignTimeScale(), "TT")
        self.assertAlmostEqual(frame.getLTOffset(), 1.1, places=3)
        self.assertAlmostEqual(frame.getTimeOrigin(), 2.2, places=3)
        self.assertEqual(frame.getTimeScale(), "TDB")

        frame.setAlignTimeScale("LMST")
        frame.setLTOffset(55.5)
        frame.setTimeOrigin(66.6)
        frame.setTimeScale("UT1")
        self.assertEqual(frame.getAlignTimeScale(), "LMST")
        self.assertAlmostEqual(frame.getLTOffset(), 55.5, places=3)
        self.assertAlmostEqual(frame.getTimeOrigin(), 66.6, places=3)
        self.assertEqual(frame.getTimeScale(), "UT1")

if __name__ == "__main__":
    unittest.main()
