from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import MappingTestCase


class TestCmpFrame(MappingTestCase):

    def test_CmpFrame(self):
        frame1 = astshim.Frame(2, "label(1)=a, label(2)=b")
        frame2 = astshim.Frame(1, "label(1)=c")
        cmpframe = astshim.CmpFrame(frame1, frame2)

        self.assertEqual(cmpframe.getNaxes(), 3)
        self.assertEqual(cmpframe.getLabel(1), "a")
        self.assertEqual(cmpframe.getLabel(2), "b")
        self.assertEqual(cmpframe.getLabel(3), "c")

        self.checkCast(cmpframe, goodType=astshim.Mapping, badType=astshim.SkyFrame)
        self.checkPersistence(cmpframe)

if __name__ == "__main__":
    unittest.main()
