from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import MappingTestCase


class TestCmpFrame(MappingTestCase):

    def test_CmpFrame(self):
        frame1 = astshim.Frame(2, "label(1)=a, label(2)=b")
        frame2 = astshim.Frame(1, "label(1)=c")
        cmpframe = astshim.CmpFrame(frame1, frame2)
        # adding to a CmpFrame increases by 1
        self.assertEqual(frame1.getRefCount(), 2)
        # adding to a CmpFrame increases by 1
        self.assertEqual(frame2.getRefCount(), 2)

        self.assertEqual(cmpframe.nAxes, 3)
        self.assertEqual(cmpframe.getLabel(1), "a")
        self.assertEqual(cmpframe.getLabel(2), "b")
        self.assertEqual(cmpframe.getLabel(3), "c")

        self.checkPersistence(cmpframe)

if __name__ == "__main__":
    unittest.main()
