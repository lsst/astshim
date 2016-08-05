from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import MappingTestCase


class TestFrameSet(MappingTestCase):

    def test_FrameSetBasics(self):
        frame = astshim.Frame(2, "Ident=base")
        frameSet = astshim.FrameSet(frame)
        self.assertIsInstance(frameSet, astshim.FrameSet)
        self.assertEqual(frameSet.getNframe(), 1)

        # Make sure the frame is deep copied
        frame.setIdent("newIdent")
        self.assertEqual(frameSet.getFrame(frameSet.BASE).getIdent(), "base")

        newFrame = astshim.Frame(2, "Ident=current")
        mapping = astshim.UnitMap(2, "Ident=mapping")
        frameSet.addFrame(1, mapping, newFrame)
        self.assertEqual(frameSet.getNframe(), 2)

        # Make sure new frame and mapping are deep copied
        newFrame.setIdent("newFrameIdent")
        mapping.setIdent("newMappingIdent")
        self.assertEqual(frameSet.getFrame(frameSet.CURRENT).getIdent(), "current")
        self.assertEqual(frameSet.getMapping().getIdent(), "mapping")

        # make sure BASE is available on the class and instance
        self.assertEqual(astshim.FrameSet.BASE, frameSet.BASE)

        baseframe = frameSet.getFrame(frameSet.BASE)
        self.assertEqual(baseframe.getIdent(), "base")
        self.assertEqual(frameSet.getBase(), 1)
        currframe = frameSet.getFrame(frameSet.CURRENT)
        self.assertEqual(currframe.getIdent(), "current")
        self.assertEqual(frameSet.getCurrent(), 2)

        mapping = frameSet.getMapping(1, 2)
        self.assertEqual(mapping.getClass(), "UnitMap")
        frameSet.remapFrame(1, astshim.UnitMap(2))
        frameSet.removeFrame(1)
        self.assertEqual(frameSet.getNframe(), 1)

        self.checkCopy(frameSet)
        self.checkPersistence(frameSet)

    def testFrameSetFrameMappingFrameConstructor(self):
        baseFrame = astshim.Frame(2, "Ident=base")
        mapping = astshim.UnitMap(2, "Ident=mapping")
        currFrame = astshim.Frame(2, "Ident=current")
        frameSet = astshim.FrameSet(baseFrame, mapping, currFrame)
        self.assertEqual(frameSet.getNframe(), 2)
        self.assertEqual(frameSet.getBase(), 1)
        self.assertEqual(frameSet.getCurrent(), 2)

        # make sure all objects were deep copied
        baseFrame.setIdent("newBase")
        mapping.setIdent("newMapping")
        currFrame.setIdent("newCurrent")
        self.assertEqual(frameSet.getFrame(frameSet.BASE).getIdent(), "base")
        self.assertEqual(frameSet.getFrame(frameSet.CURRENT).getIdent(), "current")
        self.assertEqual(frameSet.getMapping().getIdent(), "mapping")

    def test_FrameSetGetFrame(self):
        frame = astshim.Frame(2, "Ident=base")
        frameSet = astshim.FrameSet(frame)
        self.assertIsInstance(frameSet, astshim.FrameSet)
        self.assertEqual(frameSet.getNframe(), 1)

        newFrame = astshim.Frame(2, "Ident=current")
        frameSet.addFrame(1, astshim.UnitMap(2), newFrame)
        self.assertEqual(frameSet.getNframe(), 2)

        baseFrameDeep = frameSet.getFrame(astshim.FrameSet.BASE)
        baseFrameDeep.setIdent("modifiedBase")
        self.assertEqual(frameSet.getFrame(astshim.FrameSet.BASE).getIdent(), "base")

    def test_FrameSetPermutation(self):
        """Make sure permuting frame axes also affects the mapping

        If one permutes the axes of the current frame of a frame set
        *in situ* (by calling `permAxes` on the frame set itself)
        then mappings connected to that frame are also permuted
        """
        frame1 = astshim.Frame(2)
        unitMap = astshim.UnitMap(2)
        frame2 = astshim.SkyFrame()
        frameSet = astshim.FrameSet(frame1, unitMap, frame2)
        self.assertAlmostEqual(frameSet.tranForward([0, 1]), [0, 1])

        # permuting the axes of the current frame also permutes the mapping
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([0, 1]), [1, 0])

        # permuting again puts things back
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([0, 1]), [0, 1])


if __name__ == "__main__":
    unittest.main()
