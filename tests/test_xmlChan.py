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
        self.assertEqual(chan.getXmlFormat(), "NATIVE")
        self.assertEqual(chan.getXmlLength(), 0)
        self.assertEqual(chan.getXmlPrefix(), "")

        zoommap = astshim.ZoomMap(3, 2.0)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def test_XmlChanSpecifiedAttributes(self):
        sstream = astshim.StringStream()
        chan = astshim.XmlChan(
            sstream, 'XmlFormat="QUOTED", XmlLength=2000, XmlPrefix="foo"')
        self.assertEqual(chan.getXmlFormat(), "QUOTED")
        self.assertEqual(chan.getXmlLength(), 2000)
        self.assertEqual(chan.getXmlPrefix(), "foo")
        zoommap = astshim.ZoomMap(4, 1.5)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def test_XmlChanSetAttributes(self):
        sstream = astshim.StringStream()
        chan = astshim.XmlChan(sstream)
        chan.setXmlFormat("QUOTED")
        chan.setXmlLength(1500)
        chan.setXmlPrefix("test")
        self.assertEqual(chan.getXmlFormat(), "QUOTED")
        self.assertEqual(chan.getXmlLength(), 1500)
        self.assertEqual(chan.getXmlPrefix(), "test")
        zoommap = astshim.ZoomMap(1, 0.5)
        self.checkXmlPersistence(sstream=sstream, chan=chan, obj=zoommap)

    def checkXmlPersistence(self, sstream, chan, obj):
        """Check that an Ast object can be persisted and unpersisted
        """
        chan.write(obj)
        sstream.sinkToSource()
        obj_copy = chan.read()
        self.assertEqual(obj.getClassName(), obj_copy.getClassName())
        self.assertEqual(obj.show(), obj_copy.show())
        self.assertEqual(str(obj), str(obj_copy))
        self.assertEqual(repr(obj), repr(obj_copy))


if __name__ == "__main__":
    unittest.main()
