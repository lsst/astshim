from __future__ import absolute_import, division, print_function
import os.path
import unittest

import astshim
from astshim.test import ObjectTestCase

DataDir = os.path.join(os.path.dirname(__file__), "data")


class TestObject(ObjectTestCase):

    def test_XmlChanDefaultAttributes(self):
        sstream = astshim.StringStream()
        chan = astshim.XmlChan(sstream)
        self.assertEqual(chan.xmlFormat, "NATIVE")
        self.assertEqual(chan.xmlLength, 0)
        self.assertEqual(chan.xmlPrefix, "")

        zoommap = astshim.ZoomMap(3, 2.0)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def test_XmlChanSpecifiedAttributes(self):
        sstream = astshim.StringStream()
        chan = astshim.XmlChan(
            sstream, 'XmlFormat="QUOTED", XmlLength=2000, XmlPrefix="foo"')
        self.assertEqual(chan.xmlFormat, "QUOTED")
        self.assertEqual(chan.xmlLength, 2000)
        self.assertEqual(chan.xmlPrefix, "foo")
        zoommap = astshim.ZoomMap(4, 1.5)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def test_XmlChanSetAttributes(self):
        sstream = astshim.StringStream()
        chan = astshim.XmlChan(sstream)
        chan.xmlFormat = "QUOTED"
        chan.xmlLength = 1500
        chan.xmlPrefix = "test"
        self.assertEqual(chan.xmlFormat, "QUOTED")
        self.assertEqual(chan.xmlLength, 1500)
        self.assertEqual(chan.xmlPrefix, "test")
        zoommap = astshim.ZoomMap(1, 0.5)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def checkXmlPersistence(self, sstream, chan, obj):
        """Check that an Ast object can be persisted and unpersisted
        """
        chan.write(obj)
        sstream.sinkToSource()
        obj_copy = chan.read()
        self.assertEqual(obj.className, obj_copy.className)
        self.assertEqual(obj.show(), obj_copy.show())
        self.assertEqual(str(obj), str(obj_copy))
        self.assertEqual(repr(obj), repr(obj_copy))


if __name__ == "__main__":
    unittest.main()
