import unittest

import astshim as ast
from astshim.test import MappingTestCase


class TestMapSplit(MappingTestCase):

    def test_MapSplit(self):
        """Test MapSplit for a simple case"""
        nin = 3
        zoom = 1.3
        zoommap = ast.ZoomMap(nin, zoom)

        for i in range(nin):
            split = ast.MapSplit(zoommap, [i + 1])
            self.assertEqual(split.splitMap.className, "ZoomMap")
            self.assertEqual(split.splitMap.nIn, 1)
            self.assertEqual(split.splitMap.nOut, 1)
            self.assertEqual(tuple(split.origOut), (i + 1,))

        split2 = ast.MapSplit(zoommap, [1, 3])
        self.assertEqual(split2.splitMap.className, "ZoomMap")
        self.assertEqual(split2.splitMap.nIn, 2)
        self.assertEqual(split2.splitMap.nOut, 2)
        self.assertEqual(tuple(split2.origOut), (1, 3))


if __name__ == "__main__":
    unittest.main()
