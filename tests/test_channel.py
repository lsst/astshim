import filecmp
import os
import unittest

import astshim as ast
from astshim.test import MappingTestCase


class TestChannel(MappingTestCase):

    def setUp(self):
        self.dataDir = os.path.join(os.path.dirname(__file__), "data")

    def test_ChannelFileStream(self):
        path1 = os.path.join(self.dataDir, "channelFileStream1.txt")
        path2 = os.path.join(self.dataDir, "channelFileStream2.txt")

        outstream = ast.FileStream(path1, True)
        outchan = ast.Channel(outstream)
        self.assertIsInstance(outchan, ast.Object)
        self.assertIsInstance(outchan, ast.Channel)

        zoommap = ast.ZoomMap(2, 0.1, "ID=Hello there")
        nobj = outchan.write(zoommap)
        self.assertEqual(nobj, 1)

        with self.assertRaises(RuntimeError):
            obj = outchan.read()

        instream = ast.FileStream(path1, False)
        inchan = ast.Channel(instream)
        obj = inchan.read()
        self.assertEqual(obj.show(), zoommap.show())

        outstream2 = ast.FileStream(path2, True)
        outchan2 = ast.Channel(outstream2)
        outchan2.write(obj)
        self.assertTrue(filecmp.cmp(path1, path2, shallow=False))
        os.remove(path1)
        os.remove(path2)

    def test_ChannelStringStream(self):
        ss = ast.StringStream()
        channel = ast.Channel(ss)
        zoommap = ast.ZoomMap(2, 0.1, "ID=Hello there")
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
