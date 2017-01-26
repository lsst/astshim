from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import MappingTestCase


class TestFrameSet(MappingTestCase):

    def test_FrameSet(self):
        frame = astshim.Frame(2, "Ident=base")
        frameset = astshim.FrameSet(frame)
        self.assertIsInstance(frameset, astshim.FrameSet)
        self.assertEqual(frameset.getNframe(), 1)

        newframe = astshim.Frame(2, "Ident=current")
        frameset.addFrame(1, astshim.UnitMap(2), newframe)
        self.assertEqual(frameset.getNframe(), 2)

        # make sure BASE is available on the class and instance
        self.assertEqual(astshim.FrameSet.BASE, frameset.BASE)

        baseframe = frameset.getFrame(frameset.BASE)
        self.assertEqual(baseframe.getIdent(), "base")
        self.assertEqual(frameset.getBase(), 1)
        currframe = frameset.getFrame(frameset.CURRENT)
        self.assertEqual(currframe.getIdent(), "current")
        self.assertEqual(frameset.getCurrent(), 2)

        mapping = frameset.getMapping(1, 2)
        self.assertEqual(mapping.getClass(), "UnitMap")
        frameset.remapFrame(1, astshim.UnitMap(2))
        frameset.removeFrame(1)
        self.assertEqual(frameset.getNframe(), 1)

        self.checkCopy(frameset)
        self.checkPersistence(frameset)


if __name__ == "__main__":
    unittest.main()
