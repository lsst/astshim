from __future__ import absolute_import, division, print_function
import math
import unittest

import astshim
from astshim.test import MappingTestCase


class TestSkyFrame(MappingTestCase):

    def test_FrameBasics(self):
        frame = astshim.SkyFrame()
        self.assertEqual(frame.getClass(), "SkyFrame")
        self.assertEqual(frame.getNin(), 2)
        self.assertEqual(frame.getNaxes(), 2)
        self.assertEqual(frame.getMaxAxes(), 2)
        self.assertEqual(frame.getMinAxes(), 2)
        self.assertEqual(frame.getAlignSystem(), "ICRS")
        self.assertEqual(frame.getDut1(), 0.0)
        self.assertEqual(frame.getEpoch(), 2000.0)
        self.assertEqual(frame.getObsAlt(), 0.0)
        self.assertEqual(frame.getObsLat(), "N0:00:00.00")
        self.assertEqual(frame.getObsLon(), "E0:00:00.00")
        self.assertTrue(frame.getPermute())
        self.assertFalse(frame.getPreserveAxes())
        self.assertEqual(frame.getSystem(), "ICRS")
        self.assertEqual(frame.getTitle(), "ICRS coordinates")

        self.assertTrue(math.isinf(frame.getBottom(1)))
        self.assertTrue(math.isinf(frame.getTop(1)))
        self.assertGreater(frame.getTop(1), frame.getBottom(1))
        self.assertFalse(frame.getDirection(1))
        self.assertEqual(frame.getInternalUnit(1), "rad")
        self.assertEqual(frame.getNormUnit(1), "rad")
        self.assertEqual(frame.getSymbol(1), "RA")
        self.assertEqual(frame.getUnit(1), "hh:mm:ss.s")

        self.assertAlmostEqual(frame.getBottom(2), -math.pi/2)
        self.assertAlmostEqual(frame.getTop(2), math.pi/2)
        self.assertTrue(frame.getDirection(2))
        self.assertEqual(frame.getInternalUnit(2), "rad")
        self.assertEqual(frame.getNormUnit(2), "rad")
        self.assertEqual(frame.getSymbol(2), "Dec")
        self.assertEqual(frame.getUnit(2), "ddd:mm:ss")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_SkyFrameSkyOffsetMap(self):
        skyframe = astshim.SkyFrame()
        self.assertEqual(skyframe.getClass(), "SkyFrame")

        mapping = skyframe.skyOffsetMap()
        self.assertEqual(mapping.getClass(), "UnitMap")


if __name__ == "__main__":
    unittest.main()
