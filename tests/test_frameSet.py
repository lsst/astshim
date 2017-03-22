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

    def test_FrameSetPermutationSkyFrame(self):
        """Test permuting FrameSet axes using a SkyFrame

        Permuting the axes of the current frame of a frame set
        *in situ* (by calling `permAxes` on the frame set itself)
        should update the connected mappings.
        """
        # test with arbitrary values that will not be wrapped by SkyFrame
        x = 0.257
        y = 0.832
        frame1 = astshim.Frame(2)
        unitMap = astshim.UnitMap(2)
        frame2 = astshim.SkyFrame()
        frameSet = astshim.FrameSet(frame1, unitMap, frame2)
        self.assertAlmostEqual(frameSet.tranForward([x, y]), [x, y])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [x, y])

        # permuting the axes of the current frame also permutes the mapping
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([x, y]), [y, x])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [y, x])

        # permuting again puts things back
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([x, y]), [x, y])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [x, y])

    def test_FrameSetPermutationUnequal(self):
        """Test that permuting FrameSet axes with nIn != nOut

        Permuting the axes of the current frame of a frame set
        *in situ* (by calling `permAxes` on the frame set itself)
        should update the connected mappings.

        Make nIn != nOut in order to test DM-9899
        FrameSet.permAxes would fail if nIn != nOut
        """
        # Initial mapping: 3 inputs, 2 outputs: 1-1, 2-2, 3=z
        # Test using arbitrary values for x,y,z
        x = 75.1
        y = -53.2
        z = 0.123
        frame1 = astshim.Frame(3)
        permMap = astshim.PermMap([1, 2, -1], [1, 2], [z])
        frame2 = astshim.Frame(2)
        frameSet = astshim.FrameSet(frame1, permMap, frame2)
        self.assertAlmostEqual(frameSet.tranForward([x, y, z]), [x, y])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [x, y, z])

        # permuting the axes of the current frame also permutes the mapping
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([x, y, z]), [y, x])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [y, x, z])

        # permuting again puts things back
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.tranForward([x, y, z]), [x, y])
        self.assertAlmostEqual(frameSet.tranInverse([x, y]), [x, y, z])


if __name__ == "__main__":
    unittest.main()
