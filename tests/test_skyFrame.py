from __future__ import absolute_import, division, print_function
import math
import unittest

from numpy.testing import assert_allclose

import astshim
from astshim.test import MappingTestCase


class TestSkyFrame(MappingTestCase):

    def test_FrameBasics(self):
        frame = astshim.SkyFrame()
        self.assertEqual(frame.getClass(), "SkyFrame")
        self.assertEqual(frame.getNIn(), 2)
        self.assertEqual(frame.getNAxes(), 2)
        self.assertEqual(frame.getMaxAxes(), 2)
        self.assertEqual(frame.getMinAxes(), 2)

        # default values for Frame properties (methods below test
        # setters and getters of SkyFrame properties)
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

        self.assertAlmostEqual(frame.getBottom(2), -math.pi / 2)
        self.assertAlmostEqual(frame.getTop(2), math.pi / 2)
        self.assertTrue(frame.getDirection(2))
        self.assertEqual(frame.getInternalUnit(2), "rad")
        self.assertEqual(frame.getNormUnit(2), "rad")
        self.assertEqual(frame.getSymbol(2), "Dec")
        self.assertEqual(frame.getUnit(2), "ddd:mm:ss")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_SkyFrameLonLat(self):

        frame = astshim.SkyFrame()

        self.assertEqual(frame.getLonAxis(), 1)
        self.assertEqual(frame.getLatAxis(), 2)
        self.assertTrue(frame.getIsLonAxis(1))
        self.assertTrue(frame.getIsLatAxis(2))
        self.assertFalse(frame.getIsLonAxis(2))
        self.assertFalse(frame.getIsLatAxis(1))

        # permute axes
        frame.permAxes([2, 1])
        self.assertEqual(frame.getLonAxis(), 2)
        self.assertEqual(frame.getLatAxis(), 1)
        self.assertTrue(frame.getIsLonAxis(2))
        self.assertTrue(frame.getIsLatAxis(1))
        self.assertFalse(frame.getIsLonAxis(1))
        self.assertFalse(frame.getIsLatAxis(2))

        # permute again to restore original state
        frame.permAxes([2, 1])
        self.assertEqual(frame.getLonAxis(), 1)
        self.assertEqual(frame.getLatAxis(), 2)
        self.assertTrue(frame.getIsLonAxis(1))
        self.assertTrue(frame.getIsLatAxis(2))
        self.assertFalse(frame.getIsLonAxis(2))
        self.assertFalse(frame.getIsLatAxis(1))

    def test_SkyFrameAlignOffset(self):
        frame = astshim.SkyFrame()

        self.assertFalse(frame.getAlignOffset())
        frame.setAlignOffset(True)
        self.assertTrue(frame.getAlignOffset())

    def test_SkyFrameAsTime(self):
        frame = astshim.SkyFrame()

        for axis, defAsTime in ((1, True), (2, False)):
            self.assertEqual(frame.getAsTime(axis), defAsTime)
            frame.setAsTime(axis, not defAsTime)
            self.assertEqual(frame.getAsTime(axis), not defAsTime)

    def test_SkyFrameEquinox(self):
        frame = astshim.SkyFrame()

        self.assertAlmostEqual(frame.getEquinox(), 2000)
        newEquinox = 2345.6
        frame.setEquinox(newEquinox)
        self.assertAlmostEqual(frame.getEquinox(), newEquinox)

    def test_SkyFrameNegLon(self):
        frame = astshim.SkyFrame()

        self.assertFalse(frame.getNegLon())
        frame.setNegLon(True)
        self.assertTrue(frame.getNegLon())

    def test_SkyFrameProjection(self):
        frame = astshim.SkyFrame()

        self.assertEqual(frame.getProjection(), "")
        newProjection = "Arbitrary description"
        frame.setProjection(newProjection)
        self.assertEqual(frame.getProjection(), newProjection)

    def test_SkyFrameSkyRef(self):
        frame = astshim.SkyFrame()

        assert_allclose(frame.getSkyRef(), [0, 0])
        newSkyRef = [-4.5, 1.23]
        frame.setSkyRef(newSkyRef)
        assert_allclose(frame.getSkyRef(), newSkyRef)

    def test_SkyFrameSkyRefIs(self):
        frame = astshim.SkyFrame()

        self.assertEqual(frame.getSkyRefIs(), "Ignored")
        for newSkyRefIs in ("Origin", "Pole", "Ignored"):
            frame.setSkyRefIs(newSkyRefIs)
            self.assertEqual(frame.getSkyRefIs(), newSkyRefIs)

    def test_SkyFrameSkyRefP(self):
        frame = astshim.SkyFrame()

        defSkyRefP = [0.0, math.pi/2]
        assert_allclose(frame.getSkyRefP(), defSkyRefP)
        newSkyRefP = [0.1234, 0.5643]
        frame.setSkyRefP(newSkyRefP)
        assert_allclose(frame.getSkyRefP(), newSkyRefP)

    def test_SkyFrameSkyTol(self):
        frame = astshim.SkyFrame()

        # the default is arbitrary so do not to assume a specific value
        defSkyTol = frame.getSkyTol()
        newSkyTol = defSkyTol*1.2345
        frame.setSkyTol(newSkyTol)
        self.assertAlmostEqual(frame.getSkyTol(), newSkyTol)

    def test_SkyFrameSkyOffsetMap(self):
        frame = astshim.SkyFrame()

        mapping = frame.skyOffsetMap()
        self.assertEqual(mapping.getClass(), "UnitMap")


if __name__ == "__main__":
    unittest.main()
