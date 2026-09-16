import unittest

import astshim as ast
from astshim.test import MappingTestCase


class TestSpecFrame(MappingTestCase):

    def test_SpecFrameBasics(self):
        frame = ast.SpecFrame()
        self.assertEqual(frame.className, "SpecFrame")
        self.assertEqual(frame.nIn, 1)
        self.assertEqual(frame.nAxes, 1)
        self.assertEqual(frame.maxAxes, 1)
        self.assertEqual(frame.minAxes, 1)
        self.assertEqual(frame.alignSystem, "WAVE")
        self.assertEqual(frame.dut1, 0.0)
        self.assertEqual(frame.epoch, 2000.0)
        self.assertEqual(frame.obsAlt, 0.0)
        self.assertEqual(frame.obsLat, "N0:00:00.00")
        self.assertEqual(frame.obsLon, "E0:00:00.00")
        self.assertTrue(frame.permute)
        self.assertFalse(frame.preserveAxes)
        self.assertEqual(frame.system, "WAVE")
        self.assertEqual(frame.title, "Wavelength")

        self.assertGreater(abs(frame.getBottom(1)), 1e99)
        self.assertGreater(abs(frame.getTop(1)), 1e99)
        self.assertGreater(frame.getTop(1), frame.getBottom(1))
        self.assertTrue(frame.getDirection(1))
        self.assertEqual(frame.getInternalUnit(1), "Angstrom")
        # SpecFrame reports a Unit of its own while leaving the Axis Unit
        # unset, so before AST 9.5.0 NormUnit tracked the Axis, whose Unit is
        # blank. See the comment in test_frame.py about AST versions.
        if ast.astVersion() >= 9005000:
            self.assertEqual(frame.getNormUnit(1), frame.getUnit(1))
            self.assertEqual(frame.getNormUnit(1), "Angstrom")
        else:
            self.assertEqual(frame.getNormUnit(1), "")
        self.assertEqual(frame.getSymbol(1), "WAVE")
        self.assertEqual(frame.getUnit(1), "Angstrom")

        self.checkCopy(frame)
        self.checkPersistence(frame)

    def test_SpecFrameSetGetRefPos(self):
        frame = ast.SpecFrame()
        self.assertEqual(frame.className, "SpecFrame")
        sky = ast.SkyFrame()
        frame.setRefPos(sky, 0, 1)

        refpos = frame.getRefPos(sky)
        self.assertAlmostEqual(refpos[0], 0, places=5)
        self.assertAlmostEqual(refpos[1], 1)
        self.assertEqual(frame.getInternalUnit(1), frame.getUnit(1))
        self.assertEqual(frame.getInternalUnit(1), "Angstrom")


if __name__ == "__main__":
    unittest.main()
