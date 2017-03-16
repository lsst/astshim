from __future__ import absolute_import, division, print_function
import filecmp
import os
import unittest

import astshim
from astshim.test import MappingTestCase

DataDir = os.path.join(os.path.dirname(__file__))


class TestChannel(MappingTestCase):

    def test_ChannelFileStream(self):
        path1 = os.path.join(DataDir, "channelFileStream1.txt")
        path2 = os.path.join(DataDir, "channelFileStream2.txt")

        outstream = astshim.FileStream(path1, True)
        outchan = astshim.Channel(outstream)
        self.assertIsInstance(outchan, astshim.Object)
        self.assertIsInstance(outchan, astshim.Channel)

        zoommap = astshim.ZoomMap(2, 0.1, "ID=Hello there")
        nobj = outchan.write(zoommap)
        self.assertEqual(nobj, 1)

        with self.assertRaises(RuntimeError):
            obj = outchan.read()

        instream = astshim.FileStream(path1, False)
        inchan = astshim.Channel(instream)
        obj = inchan.read()
        self.assertEqual(obj.show(), zoommap.show())

        outstream2 = astshim.FileStream(path2, True)
        outchan2 = astshim.Channel(outstream2)
        outchan2.write(obj)
        self.assertTrue(filecmp.cmp(path1, path2, shallow=False))
        os.remove(path1)
        os.remove(path2)

    def test_ChannelStringStream(self):
        ss = astshim.StringStream()
        channel = astshim.Channel(ss)
        zoommap = astshim.ZoomMap(2, 0.1, "ID=Hello there")
        n = channel.write(zoommap)
        self.assertEqual(n, 1)
        sinkData1 = ss.getSinkData()

        ss.sinkToSource()
        obj = channel.read()
        self.assertEqual(obj.show(), zoommap.show())
        n = channel.write(obj)
        self.assertEqual(n, 1)
        sinkData2 = ss.getSinkData()
        self.assertEqual(sinkData1, sinkData2)


if __name__ == "__main__":
    unittest.main()
