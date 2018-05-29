from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose

import astshim as ast
from astshim.test import MappingTestCase


class TestFrameSet(MappingTestCase):

    def test_FrameSetBasics(self):
        frame = ast.Frame(2, "Ident=base")
        initialNumFrames = frame.getNObject()  # may be >1 when run using pytest
        frameSet = ast.FrameSet(frame)
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertEqual(frameSet.nFrame, 1)

        # Make sure the frame is deep copied
        frame.ident = "newIdent"
        self.assertEqual(frameSet.getFrame(frameSet.BASE).ident, "base")
        self.assertEqual(frame.getRefCount(), 1)
        self.assertEqual(frame.getNObject(), initialNumFrames + 1)

        # add a new frame and mapping; make sure they are deep copied
        newFrame = ast.Frame(2, "Ident=current")
        mapping = ast.UnitMap(2, "Ident=mapping")
        initialNumUnitMap = mapping.getNObject()
        self.assertEqual(frame.getNObject(), initialNumFrames + 2)
        frameSet.addFrame(1, mapping, newFrame)
        self.assertEqual(frameSet.nFrame, 2)
        newFrame.ident = "newFrameIdent"
        mapping.ident = "newMappingIdent"
        self.assertEqual(frameSet.getFrame(frameSet.CURRENT).ident, "current")
        self.assertEqual(frameSet.getMapping().ident, "mapping")
        self.assertEqual(newFrame.getRefCount(), 1)
        self.assertEqual(frame.getNObject(), initialNumFrames + 3)
        self.assertEqual(mapping.getRefCount(), 1)
        self.assertEqual(mapping.getNObject(), initialNumUnitMap + 1)

        # make sure BASE is available on the class and instance
        self.assertEqual(ast.FrameSet.BASE, frameSet.BASE)

        baseframe = frameSet.getFrame(frameSet.BASE)
        self.assertEqual(frame.getNObject(), initialNumFrames + 4)
        self.assertEqual(baseframe.ident, "base")
        self.assertEqual(frameSet.base, 1)
        currframe = frameSet.getFrame(frameSet.CURRENT)
        self.assertEqual(frame.getNObject(), initialNumFrames + 5)
        self.assertEqual(currframe.ident, "current")
        self.assertEqual(frameSet.current, 2)

        self.checkCopy(frameSet)

        input_data = np.array([
            [0.0, 0.1, -1.5],
            [5.1, 0.0, 3.1],
        ])
        self.checkMappingPersistence(frameSet, input_data)

    def testFrameSetFrameMappingFrameConstructor(self):
        baseFrame = ast.Frame(2, "Ident=base")
        mapping = ast.UnitMap(2, "Ident=mapping")
        currFrame = ast.Frame(2, "Ident=current")
        frameSet = ast.FrameSet(baseFrame, mapping, currFrame)
        self.assertEqual(frameSet.nFrame, 2)
        self.assertEqual(frameSet.base, 1)
        self.assertEqual(frameSet.current, 2)

        # make sure all objects were deep copied
        baseFrame.ident = "newBase"
        mapping.ident = "newMapping"
        currFrame.ident = "newCurrent"
        self.assertEqual(frameSet.getFrame(frameSet.BASE).ident, "base")
        self.assertEqual(frameSet.getFrame(frameSet.CURRENT).ident, "current")
        self.assertEqual(frameSet.getMapping().ident, "mapping")

    def test_FrameSetGetFrame(self):
        frame = ast.Frame(2, "Ident=base")
        frameSet = ast.FrameSet(frame)
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertEqual(frameSet.nFrame, 1)

        newFrame = ast.Frame(2, "Ident=current")
        frameSet.addFrame(1, ast.UnitMap(2), newFrame)
        self.assertEqual(frameSet.nFrame, 2)

        # check that getFrame returns a deep copy
        baseFrameDeep = frameSet.getFrame(ast.FrameSet.BASE)
        self.assertEqual(baseFrameDeep.ident, "base")
        self.assertEqual(baseFrameDeep.getRefCount(), 1)
        baseFrameDeep.ident = "modifiedBase"
        self.assertEqual(frameSet.getFrame(ast.FrameSet.BASE).ident, "base")
        self.assertEqual(frame.ident, "base")

    def test_FrameSetGetMapping(self):
        frame = ast.Frame(2, "Ident=base")
        frameSet = ast.FrameSet(frame)
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertEqual(frameSet.nFrame, 1)

        newFrame = ast.Frame(2)
        mapping = ast.UnitMap(2, "Ident=mapping")
        initialNumUnitMap = mapping.getNObject()  # may be >1 when run using pytest
        frameSet.addFrame(1, mapping, newFrame)
        self.assertEqual(frameSet.nFrame, 2)
        self.assertEqual(mapping.getNObject(), initialNumUnitMap + 1)

        # check that getMapping returns a deep copy
        mappingDeep = frameSet.getMapping(1, 2)
        self.assertEqual(mappingDeep.ident, "mapping")
        mappingDeep.ident = "modifiedMapping"
        self.assertEqual(mapping.ident, "mapping")
        self.assertEqual(mappingDeep.getRefCount(), 1)
        self.assertEqual(mapping.getNObject(), initialNumUnitMap + 2)

    def test_FrameSetRemoveFrame(self):
        frame = ast.Frame(2, "Ident=base")
        initialNumFrames = frame.getNObject()  # may be >1 when run using pytest
        frameSet = ast.FrameSet(frame)
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertEqual(frameSet.nFrame, 1)
        self.assertEqual(frame.getNObject(), initialNumFrames + 1)

        newFrame = ast.Frame(2, "Ident=current")
        self.assertEqual(frame.getNObject(), initialNumFrames + 2)
        zoomMap = ast.ZoomMap(2, 0.5, "Ident=zoom")
        initialNumZoomMap = zoomMap.getNObject()
        frameSet.addFrame(1, zoomMap, newFrame)
        self.assertEqual(frameSet.nFrame, 2)
        self.assertEqual(frame.getNObject(), initialNumFrames + 3)
        self.assertEqual(zoomMap.getNObject(), initialNumZoomMap + 1)

        # remove the frame named "base", leaving the frame named "current"
        frameSet.removeFrame(1)
        self.assertEqual(frameSet.nFrame, 1)
        # removing one frame leaves frame, newFrame and a copy of newFrame in FrameSet
        self.assertEqual(frame.getNObject(), initialNumFrames + 2)
        self.assertEqual(zoomMap.getNObject(), initialNumZoomMap)
        frameDeep = frameSet.getFrame(1)
        self.assertEqual(frameDeep.ident, "current")

        # it is not allowed to remove the last frame
        with self.assertRaises(RuntimeError):
            frameSet.removeFrame(1)

    def test_FrameSetRemapFrame(self):
        frame = ast.Frame(2, "Ident=base")
        initialNumFrames = frame.getNObject()  # may be >1 when run using pytest
        frameSet = ast.FrameSet(frame)
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertEqual(frameSet.nFrame, 1)
        self.assertEqual(frame.getNObject(), initialNumFrames + 1)

        newFrame = ast.Frame(2, "Ident=current")
        self.assertEqual(frame.getNObject(), initialNumFrames + 2)
        zoom = 0.5
        zoomMap = ast.ZoomMap(2, zoom, "Ident=zoom")
        initialNumZoomMap = zoomMap.getNObject()
        frameSet.addFrame(1, zoomMap, newFrame)
        self.assertEqual(frameSet.nFrame, 2)
        self.assertEqual(frame.getNObject(), initialNumFrames + 3)
        self.assertEqual(zoomMap.getNObject(), initialNumZoomMap + 1)

        input_data = np.array([
            [0.0, 0.1, -1.5],
            [5.1, 0.0, 3.1],
        ])
        predicted_output1 = input_data * zoom
        assert_allclose(frameSet.applyForward(input_data), predicted_output1, atol=1e-12)
        self.checkMappingPersistence(frameSet, input_data)

        shift = (0.5, -1.5)
        shiftMap = ast.ShiftMap(shift, "Ident=shift")
        initialNumShiftMap = shiftMap.getNObject()
        self.assertEqual(zoomMap.getNObject(), initialNumZoomMap + 1)
        frameSet.remapFrame(1, shiftMap)
        self.assertEqual(zoomMap.getNObject(), initialNumZoomMap + 1)
        self.assertEqual(shiftMap.getNObject(), initialNumShiftMap + 1)
        predicted_output2 = (input_data.T - shift).T * zoom
        assert_allclose(frameSet.applyForward(input_data), predicted_output2, atol=1e-12)

    def test_FrameSetPermutationSkyFrame(self):
        """Test permuting FrameSet axes using a SkyFrame

        Permuting the axes of the current frame of a frame set
        *in situ* (by calling `permAxes` on the frame set itself)
        should update the connected mappings.
        """
        # test with arbitrary values that will not be wrapped by SkyFrame
        x = 0.257
        y = 0.832
        frame1 = ast.Frame(2)
        unitMap = ast.UnitMap(2)
        frame2 = ast.SkyFrame()
        frameSet = ast.FrameSet(frame1, unitMap, frame2)
        self.assertAlmostEqual(frameSet.applyForward([x, y]), [x, y])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [x, y])

        # permuting the axes of the current frame also permutes the mapping
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.applyForward([x, y]), [y, x])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [y, x])

        # permuting again puts things back
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.applyForward([x, y]), [x, y])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [x, y])

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
        frame1 = ast.Frame(3)
        permMap = ast.PermMap([1, 2, -1], [1, 2], [z])
        frame2 = ast.Frame(2)
        frameSet = ast.FrameSet(frame1, permMap, frame2)
        self.assertAlmostEqual(frameSet.applyForward([x, y, z]), [x, y])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [x, y, z])

        # permuting the axes of the current frame also permutes the mapping
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.applyForward([x, y, z]), [y, x])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [y, x, z])

        # permuting again puts things back
        frameSet.permAxes([2, 1])
        self.assertAlmostEqual(frameSet.applyForward([x, y, z]), [x, y])
        self.assertAlmostEqual(frameSet.applyInverse([x, y]), [x, y, z])


if __name__ == "__main__":
    unittest.main()
