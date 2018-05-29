from __future__ import absolute_import, division, print_function
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestFrame(MappingTestCase):

    def test_FrameBasics(self):
        frame = ast.Frame(2)
        self.assertEqual(frame.className, "Frame")
        self.assertEqual(frame.nIn, 2)
        self.assertEqual(frame.nAxes, 2)
        self.assertEqual(frame.maxAxes, 2)
        self.assertEqual(frame.minAxes, 2)
        self.assertEqual(frame.alignSystem, "Cartesian")
        self.assertEqual(frame.dut1, 0.0)
        self.assertEqual(frame.epoch, 2000.0)
        self.assertEqual(frame.obsAlt, 0.0)
        self.assertEqual(frame.obsLat, "N0:00:00.00")
        self.assertEqual(frame.obsLon, "E0:00:00.00")
        self.assertTrue(frame.permute)
        self.assertFalse(frame.preserveAxes)
        self.assertEqual(frame.system, "Cartesian")
        self.assertEqual(frame.title, "2-d coordinate system")
        self.assertEqual(frame.getDigits(), 7)

        for axis in (1, 2):
            self.assertGreater(abs(frame.getBottom(axis)), 1e99)
            self.assertGreater(abs(frame.getTop(axis)), 1e99)
            self.assertGreater(frame.getTop(axis), frame.getBottom(axis))
            self.assertTrue(frame.getDirection(axis))
            self.assertEqual(frame.getDigits(axis), 7)
            self.assertEqual(frame.getInternalUnit(axis), "")
            self.assertEqual(frame.getNormUnit(axis), "")
            self.assertEqual(frame.getSymbol(axis), "x{}".format(axis))
            self.assertEqual(frame.getUnit(axis), "")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_FrameSetDigits(self):
        frame = ast.Frame(2)
        self.assertEqual(frame.getDigits(), 7)
        for axis in (1, 2):
            self.assertEqual(frame.getDigits(axis), 7)

        frame.setDigits(1, 9)
        self.assertEqual(frame.getDigits(), 7)
        self.assertEqual(frame.getDigits(1), 9)
        self.assertEqual(frame.getDigits(2), 7)

        frame.setDigits(2, 4)
        self.assertEqual(frame.getDigits(), 7)
        self.assertEqual(frame.getDigits(1), 9)
        self.assertEqual(frame.getDigits(2), 4)

    def test_FrameLabels(self):
        frame = ast.Frame(2, "label(1)=a b,label(2)=c d")

        self.assertEqual(frame.getLabel(1), "a b")
        self.assertEqual(frame.getLabel(2), "c d")
        frame.setLabel(2, "A new label")
        self.assertEqual(frame.getLabel(2), "A new label")
        frame.clear("Label(2)")
        self.assertEqual(frame.getLabel(2), "Axis 2")

    def test_FrameTitle(self):
        frame = ast.Frame(3, "Title=A Title")

        self.assertEqual(frame.title, "A Title")
        testtitle = "Test Frame"
        frame.title = testtitle
        frame.clear("Title")
        self.assertEqual(frame.title, "3-d coordinate system")

    def test_FrameAngle(self):
        """Test Frame.angle"""
        frame = ast.Frame(2)
        angle = frame.angle([4, 3], [0, 0], [4, 0])
        self.assertEqual(angle, math.atan2(3, 4))

    def test_FrameAxis(self):
        """Test Frame.axAngle, axDistance and axOffset"""
        frame = ast.Frame(2)
        angle = frame.axAngle([0, 0], [4, 3], 1)
        self.assertEqual(angle, -math.atan2(3, 4))
        distance = frame.axDistance(1, 0, 4)
        self.assertEqual(distance, 4)
        axoffset = frame.axOffset(1, 1, 4)
        self.assertEqual(axoffset, 5)

    def test_FrameConvert(self):
        frame = ast.Frame(2)
        nframe = ast.Frame(2)
        fset = frame.convert(nframe)
        self.assertEqual(fset.className, "FrameSet")

        # the conversion FrameSet should contain two frames
        # connected by a unit mapping with 2 axes
        self.assertEqual(fset.nFrame, 2)
        self.assertEqual(fset.nIn, 2)
        self.assertEqual(fset.nOut, 2)
        indata = np.array([
            [1.1, 2.2],
            [-43.5, 1309.31],
        ])
        outdata = fset.applyForward(indata)
        assert_allclose(outdata, indata, atol=1e-12)
        self.checkRoundTrip(fset, indata)

        self.assertIsNone(frame.convert(ast.Frame(3)))

    def test_FrameFindFrame(self):
        frame = ast.Frame(2)
        nframe = ast.Frame(2)
        fset = frame.findFrame(nframe)
        self.assertEqual(fset.className, "FrameSet")
        self.assertEqual(fset.nFrame, 2)

        # the found FrameSet should contain two frames
        # connected by a unit mapping with 2 axes
        self.assertEqual(fset.nIn, 2)
        self.assertEqual(fset.nOut, 2)
        indata = np.array([
            [1.1, 2.2],
            [-43.5, 1309.31],
        ])
        outdata = fset.applyForward(indata)
        assert_allclose(outdata, indata, atol=1e-12)
        self.checkRoundTrip(fset, indata)

        self.assertIsNone(frame.findFrame(ast.Frame(3)))

    def test_FrameDistance(self):
        frame = ast.Frame(2)
        distance = frame.distance([0, 0], [4, 3])
        self.assertEqual(distance, 5)

    def test_FrameFormat(self):
        frame = ast.Frame(2)
        fmt = frame.format(1, 55.270)
        self.assertEqual(fmt, "55.27")

    def test_FrameIntersect(self):
        frame = ast.Frame(2)
        cross = frame.intersect([-1, 1], [1, 1], [0, 0], [2, 2])
        self.assertAlmostEqual(cross[0], 1.0)
        self.assertAlmostEqual(cross[1], 1.0)

    def test_FrameMatchAxes(self):
        frame = ast.Frame(2)
        frame2 = ast.Frame(3)
        axes = frame.matchAxes(frame2)
        self.assertEqual(axes[0], 1)
        self.assertEqual(axes[1], 2)
        self.assertEqual(axes[2], 0)

    def test_FrameNorm(self):
        frame = ast.Frame(2)
        # arbitrary, but large enough to wrap if applied to an SphFrame
        coords = [33.5, 223.4]
        ncoords = frame.norm(coords)
        self.assertEqual(ncoords[0], coords[0])

    def test_FrameOffset(self):
        """Test Frame.offset and Frame.offset2"""
        frame = ast.Frame(2)
        point = frame.offset([0, 0], [4, 3], 10)
        self.assertEqual(point[0], 8)
        self.assertEqual(point[1], 6)
        dp = frame.offset2([0, 0], math.atan2(4, 3), 10)
        self.assertAlmostEqual(dp.point[0], 8)
        self.assertAlmostEqual(dp.point[1], 6)

    def test_FrameOver(self):
        frame1 = ast.Frame(2, "label(1)=a, label(2)=b")
        initialNumFrames = frame1.getNObject()  # may be >1 when run using pytest
        frame2 = ast.Frame(1, "label(1)=c")
        self.assertEqual(frame1.getNObject(), initialNumFrames + 1)
        cf = frame1.under(frame2)
        self.assertEqual(cf.nAxes, 3)
        self.assertEqual(cf.getLabel(1), "a")
        self.assertEqual(cf.getLabel(2), "b")
        self.assertEqual(cf.getLabel(3), "c")

        # check that the contained frames are shallow copies
        self.assertEqual(frame1.getNObject(), initialNumFrames + 1)
        self.assertEqual(frame1.getRefCount(), 2)
        self.assertEqual(frame2.getRefCount(), 2)

    def test_FramePerm(self):
        frame = ast.Frame(2)
        frame.permAxes([2, 1])
        fm = frame.pickAxes([2])
        self.assertEqual(fm.frame.className, "Frame")
        self.assertEqual(fm.frame.nIn, 1)
        self.assertEqual(fm.mapping.className, "PermMap")
        self.assertEqual(fm.mapping.nIn, 2)
        self.assertEqual(fm.mapping.nOut, 1)

    def test_FrameResolve(self):
        frame = ast.Frame(2)
        res = frame.resolve([0, 0], [2, 1], [0, 4])
        theta = math.atan2(1, 2)
        d1pred = 4 * math.sin(theta)
        d2pred = 4 * math.cos(theta)
        predpoint = [
            d1pred * math.cos(theta),
            d1pred * math.sin(theta),
        ]
        self.assertAlmostEqual(res.d1, d1pred)
        self.assertAlmostEqual(res.d2, d2pred)
        assert_allclose(res.point, predpoint, atol=1e-12)

    def test_FrameUnformat(self):
        frame = ast.Frame(2)
        nrv = frame.unformat(1, "56.4 #")
        self.assertEqual(nrv.nread, 5)
        self.assertEqual(nrv.value, 56.4)

    def test_FrameActiveUnit(self):
        """Test the ActiveUnit property"""
        frame = ast.Frame(2)
        self.assertFalse(frame.activeUnit)
        frame.activeUnit = True
        self.assertTrue(frame.activeUnit)


if __name__ == "__main__":
    unittest.main()
