from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np

import astshim
from astshim.test import MappingTestCase


class TestFrame(MappingTestCase):

    def test_FrameBasics(self):
        frame = astshim.Frame(2)
        self.assertEqual(frame.getClass(), "Frame")
        self.assertEqual(frame.getNin(), 2)
        self.assertEqual(frame.getNaxes(), 2)
        self.assertEqual(frame.getMaxAxes(), 2)
        self.assertEqual(frame.getMinAxes(), 2)
        self.assertEqual(frame.getAlignSystem(), "Cartesian")
        self.assertEqual(frame.getDut1(), 0.0)
        self.assertEqual(frame.getEpoch(), 2000.0)
        self.assertEqual(frame.getObsAlt(), 0.0)
        self.assertEqual(frame.getObsLat(), "N0:00:00.00")
        self.assertEqual(frame.getObsLon(), "E0:00:00.00")
        self.assertTrue(frame.getPermute())
        self.assertFalse(frame.getPreserveAxes())
        self.assertEqual(frame.getSystem(), "Cartesian")
        self.assertEqual(frame.getTitle(), "2-d coordinate system")

        for axis in (1, 2):
            self.assertTrue(math.isinf(frame.getBottom(axis)))
            self.assertTrue(math.isinf(frame.getTop(axis)))
            self.assertGreater(frame.getTop(axis), frame.getBottom(axis))
            self.assertTrue(frame.getDirection(axis))
            self.assertEqual(frame.getInternalUnit(axis), "")
            self.assertEqual(frame.getNormUnit(axis), "")
            self.assertEqual(frame.getSymbol(axis), "x{}".format(axis))
            self.assertEqual(frame.getUnit(axis), "")

        self.checkCast(frame, goodType=astshim.Mapping, badType=astshim.CmpMap)
        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_FrameLabels(self):
        frame = astshim.Frame(2, "label(1)=a b,label(2)=c d")

        self.assertEqual(frame.getLabel(1), "a b")
        self.assertEqual(frame.getLabel(2), "c d")
        frame.setLabel(2, "A new label")
        self.assertEqual(frame.getLabel(2), "A new label")
        frame.clear("Label(2)")
        self.assertEqual(frame.getLabel(2), "Axis 2")

    def test_FrameTitle(self):
        frame = astshim.Frame(3, "Title=A Title")

        self.assertEqual(frame.getTitle(), "A Title")
        testtitle = "Test Frame"
        frame.setTitle(testtitle)
        frame.clear("Title")
        self.assertEqual(frame.getTitle(), "3-d coordinate system")

    def test_FrameAngle(self):
        """Test Frame.angle"""
        frame = astshim.Frame(2)
        angle = frame.angle([4, 3], [0, 0], [4, 0])
        self.assertEqual(angle, math.atan2(3, 4))

    def test_FrameAxis(self):
        """Test Frame.axAngle, axDistance and axOffset"""
        frame = astshim.Frame(2)
        angle = frame.axAngle([0, 0], [4, 3], 1)
        self.assertEqual(angle, -math.atan2(3, 4))
        distance = frame.axDistance(1, 0, 4)
        self.assertEqual(distance, 4)
        axoffset = frame.axOffset(1, 1, 4)
        self.assertEqual(axoffset, 5)

    def test_FrameConvert(self):
        frame = astshim.Frame(2)
        nframe = astshim.Frame(2)
        fset = frame.convert(nframe)
        self.assertEqual(fset.getClass(), "FrameSet")
        self.assertEqual(fset.getNframe(), 2)
        fset2 = fset.findFrame(nframe)
        self.assertEqual(fset2.getClass(), "FrameSet")

    def test_FrameDistance(self):
        frame = astshim.Frame(2)
        distance = frame.distance([0, 0], [4, 3])
        self.assertEqual(distance, 5)

    def test_FrameFormat(self):
        frame = astshim.Frame(2)
        fmt = frame.format(1, 55.270)
        self.assertEqual(fmt, "55.27")

    def test_FrameIntersect(self):
        frame = astshim.Frame(2)
        cross = frame.intersect([-1, 1], [1, 1], [0, 0], [2, 2])
        self.assertAlmostEqual(cross[0], 1.0)
        self.assertAlmostEqual(cross[1], 1.0)

    def test_FrameMatchAxes(self):
        frame = astshim.Frame(2)
        frame2 = astshim.Frame(3)
        axes = frame.matchAxes(frame2)
        self.assertEqual(axes[0], 1)
        self.assertEqual(axes[1], 2)
        self.assertEqual(axes[2], 0)

    def test_FrameNorm(self):
        frame = astshim.Frame(2)
        coords = [33.5, 223.4]  # arbitrary, but large enough to wrap if applied to an SphFrame
        ncoords = frame.norm(coords)
        self.assertEqual(ncoords[0], coords[0])

    def test_FrameOffset(self):
        """Test Frame.offset and Frame.offset2"""
        frame = astshim.Frame(2)
        point = frame.offset([0, 0], [4, 3], 10)
        self.assertEqual(point[0], 8)
        self.assertEqual(point[1], 6)
        dp = frame.offset2([0, 0], math.atan2(4, 3), 10)
        self.assertAlmostEqual(dp.point[0], 8)
        self.assertAlmostEqual(dp.point[1], 6)

    def test_FrameOver(self):
        frame1 = astshim.Frame(2, "label(1)=a, label(2)=b")
        frame2 = astshim.Frame(1, "label(1)=c")
        cf = frame2.over(frame1)
        self.assertEqual(cf.getNaxes(), 3)
        self.assertEqual(cf.getLabel(1), "a")
        self.assertEqual(cf.getLabel(2), "b")
        self.assertEqual(cf.getLabel(3), "c")

    def test_FramePerm(self):
        frame = astshim.Frame(2)
        frame.permAxes([2, 1])
        fm = frame.pickAxes([2])
        self.assertEqual(fm.frame.getClass(), "Frame")
        self.assertEqual(fm.frame.getNin(), 1)
        self.assertEqual(fm.mapping.getClass(), "PermMap")
        self.assertEqual(fm.mapping.getNin(), 2)
        self.assertEqual(fm.mapping.getNout(), 1)

    def test_FrameResolve(self):
        frame = astshim.Frame(2)
        res = frame.resolve([0, 0], [2, 1], [0, 4])
        theta = math.atan2(1, 2)
        d1pred = 4*math.sin(theta)
        d2pred = 4*math.cos(theta)
        predpoint = [
            d1pred*math.cos(theta),
            d1pred*math.sin(theta),
        ]
        self.assertAlmostEqual(res.d1, d1pred)
        self.assertAlmostEqual(res.d2, d2pred)
        self.assertTrue(np.allclose(res.point, predpoint))

    def test_FrameUnformat(self):
        frame = astshim.Frame(2)
        nrv = frame.unformat(1, "56.4 #")
        self.assertEqual(nrv.nread, 5)
        self.assertEqual(nrv.value, 56.4)

    def test_FrameActiveUnit(self):
        """Test the ActiveUnit property"""
        frame = astshim.Frame(2)
        self.assertFalse(frame.getActiveUnit())
        frame.setActiveUnit(True)
        self.assertTrue(frame.getActiveUnit())


if __name__ == "__main__":
    unittest.main()
