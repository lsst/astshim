from __future__ import absolute_import, division, print_function
import math
import unittest

import astshim
from astshim.test import MappingTestCase


class TestSpecFrame(MappingTestCase):

    def test_SpecFrameBasics(self):
        frame = astshim.SpecFrame()
        self.assertEqual(frame.getClass(), "SpecFrame")
        self.assertEqual(frame.getNin(), 1)
        self.assertEqual(frame.getNaxes(), 1)
        self.assertEqual(frame.getMaxAxes(), 1)
        self.assertEqual(frame.getMinAxes(), 1)
        self.assertEqual(frame.getAlignSystem(), "WAVE")
        self.assertEqual(frame.getDut1(), 0.0)
        self.assertEqual(frame.getEpoch(), 2000.0)
        self.assertEqual(frame.getObsAlt(), 0.0)
        self.assertEqual(frame.getObsLat(), "N0:00:00.00")
        self.assertEqual(frame.getObsLon(), "E0:00:00.00")
        self.assertTrue(frame.getPermute())
        self.assertFalse(frame.getPreserveAxes())
        self.assertEqual(frame.getSystem(), "WAVE")
        self.assertEqual(frame.getTitle(), "Wavelength")

        self.assertTrue(math.isinf(frame.getBottom(1)))
        self.assertTrue(math.isinf(frame.getTop(1)))
        self.assertGreater(frame.getTop(1), frame.getBottom(1))
        self.assertTrue(frame.getDirection(1))
        self.assertEqual(frame.getInternalUnit(1), "Angstrom")
        self.assertEqual(frame.getNormUnit(1), "")
        self.assertEqual(frame.getSymbol(1), "WAVE")
        self.assertEqual(frame.getUnit(1), "Angstrom")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_SpecFrameSetGetRefPos(self):
        frame = astshim.SpecFrame()
        self.assertEqual(frame.getClass(), "SpecFrame")
        sky = astshim.SkyFrame()
        frame.setRefPos(sky, 0, 1)

        refpos = frame.getRefPos(sky)
        self.assertAlmostEqual(refpos[0], 0, places=5)
        self.assertAlmostEqual(refpos[1], 1)
        self.assertEqual(frame.getInternalUnit(1), frame.getUnit(1))
        self.assertEqual(frame.getInternalUnit(1), "Angstrom")


if __name__ == "__main__":
    unittest.main()
