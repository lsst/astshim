from __future__ import absolute_import, division, print_function
import unittest
from numpy.testing import assert_allclose

from astshim import Frame, SkyFrame, UnitMap, FrameSet, append
from astshim.test import makeForwardPolyMap, makeTwoWayPolyMap


class TestFrameSetAppend(unittest.TestCase):

    def makeFrameSet(self, nIn, nOut):
        """Create a FrameSet with the specified dimensions.

        The FrameSet shall have the following Frames:
              "base"(nIn, #1)
                    |
                    +--------------
                    |             |
               "mid"(4, #2)  "fork"(1, #4)
                    |
            "current"(nOut, #3)
        """
        frameSet = FrameSet(Frame(nIn, "Ident=base"))
        frameSet.addFrame(FrameSet.CURRENT,
                          makeTwoWayPolyMap(nIn, 4),
                          Frame(4, "Ident=mid"))
        frameSet.addFrame(FrameSet.CURRENT,
                          makeTwoWayPolyMap(4, nOut),
                          Frame(nOut, "Ident=current"))
        frameSet.addFrame(FrameSet.BASE,
                          makeTwoWayPolyMap(nIn, 1),
                          Frame(1, "Ident=fork"))

        frameSet.current = 3
        assert frameSet.nFrame == 4
        assert frameSet.base == 1
        assert frameSet.getFrame(FrameSet.BASE).ident == "base"
        assert frameSet.getFrame(2).ident == "mid"
        assert frameSet.current == 3
        assert frameSet.getFrame(FrameSet.CURRENT).ident == "current"
        assert frameSet.getFrame(4).ident == "fork"
        return frameSet

    def test_AppendEffect(self):
        """Check that a concatenated FrameSet transforms correctly.
        """
        set1 = self.makeFrameSet(2, 3)
        set2 = self.makeFrameSet(3, 1)
        set12 = append(set1, set2)

        # Can't match 1D output to 2D input
        with self.assertRaises(RuntimeError):
            append(set2, set1)

        x = [1.2, 3.4]
        y_merged = set12.tranForward(x)
        y_separate = set2.tranForward(set1.tranForward(x))
        assert_allclose(y_merged, y_separate)

        y = [-0.3]
        x_merged = set12.tranInverse(y)
        x_separate = set1.tranInverse(set2.tranInverse(y))
        assert_allclose(x_merged, x_separate)

        # No side effects
        self.assertEqual(set1.base, 1)
        self.assertEqual(set1.current, 3)
        self.assertEqual(set2.base, 1)
        self.assertEqual(set2.current, 3)

    def test_AppendFrames(self):
        """Check that a concatenated FrameSet preserves all Frames.
        """
        set1 = self.makeFrameSet(1, 3)
        # AST docs say FrameSets always have contiguously numbered frames,
        # but let's make sure
        set1.removeFrame(2)
        set2 = self.makeFrameSet(3, 2)
        set12 = append(set1, set2)

        self.assertEquals(set1.nFrame + set2.nFrame,
                          set12.nFrame)
        for i in range(1, 1+set1.nFrame):
            oldFrame = set1.getFrame(i)
            newFrame = set12.getFrame(i)
            self.assertEquals(oldFrame.ident, newFrame.ident)
            self.assertEquals(oldFrame.nAxes, newFrame.nAxes)
            if i == set1.base:
                self.assertTrue(i == set12.base)
            else:
                self.assertFalse(i == set12.base)
        for i in range(1, 1+set2.nFrame):
            offset = set1.nFrame
            oldFrame = set2.getFrame(i)
            newFrame = set12.getFrame(offset + i)
            self.assertEquals(oldFrame.ident, newFrame.ident)
            self.assertEquals(oldFrame.nAxes, newFrame.nAxes)
            if i == set2.current:
                self.assertTrue(offset + i == set12.current)
            else:
                self.assertFalse(offset + i == set12.current)

    def test_AppendIndependent(self):
        """Check that a concatenated FrameSet is not affected by changes
        to its constituents.
        """
        set1 = self.makeFrameSet(3, 3)
        set2 = self.makeFrameSet(3, 3)
        set12 = append(set1, set2)

        nTotal = set12.nFrame
        x = [1.2, 3.4, 5.6]
        y = set12.tranForward(x)

        set1.addFrame(2, makeTwoWayPolyMap(4, 2), Frame(2, "Ident=extra"))
        set1.addFrame(1, makeTwoWayPolyMap(3, 3), Frame(3, "Ident=legume"))
        set1.removeFrame(3)
        set2.addFrame(4, makeForwardPolyMap(1, 4), Frame(4, "Ident=extra"))
        set2.base = 2

        # Use exact equality because nothing should change
        self.assertEquals(set12.nFrame, nTotal)
        self.assertEquals(set12.tranForward(x), y)

    def test_AppendMismatch(self):
        """Check that append behaves as expected when joining non-identical frames.
        """
        set1 = self.makeFrameSet(3, 2)
        set2 = self.makeFrameSet(2, 3)
        set1.addFrame(FrameSet.CURRENT, makeForwardPolyMap(2, 2),
                      SkyFrame("Ident=sky"))
        set12 = append(set1, set2)

        x = [1.2, 3.4, 5.6]
        y_merged = set12.tranForward(x)
        y_separate = set2.tranForward(set1.tranForward(x))
        assert_allclose(y_merged, y_separate)

        iFrom = set1.current
        iTo = set1.nFrame + set2.base
        self.assertIsInstance(set12.getFrame(iFrom), SkyFrame)
        self.assertNotIsInstance(set12.getFrame(iTo), SkyFrame)
        self.assertIsInstance(set12.getMapping(iFrom, iTo), UnitMap)


if __name__ == "__main__":
    unittest.main()
