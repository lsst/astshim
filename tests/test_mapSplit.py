from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import MappingTestCase


class TestMapSplit(MappingTestCase):

    def test_MapSplit(self):
        """Test MapSplit for a simple case"""
        nin = 3
        zoom = 1.3
        zoommap = astshim.ZoomMap(nin, zoom)

        for i in range(nin):
            split = astshim.MapSplit(zoommap, [i+1])
            self.assertEqual(split.splitMap.getClass(), "ZoomMap")
            self.assertEqual(split.splitMap.getNin(), 1)
            self.assertEqual(split.splitMap.getNout(), 1)
            self.assertEqual(tuple(split.origOut), (i+1,))

        split2 = astshim.MapSplit(zoommap, [1, 3])
        self.assertEqual(split2.splitMap.getClass(), "ZoomMap")
        self.assertEqual(split2.splitMap.getNin(), 2)
        self.assertEqual(split2.splitMap.getNout(), 2)
        self.assertEqual(tuple(split2.origOut), (1, 3))

if __name__ == "__main__":
    unittest.main()
