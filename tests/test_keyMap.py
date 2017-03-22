from __future__ import absolute_import, division, print_function
import unittest

from numpy.testing import assert_allclose

import astshim
from astshim.test import ObjectTestCase


class TestKeyMap(ObjectTestCase):

    def test_KeyMapKey(self):
        keyMap = astshim.KeyMap("SortBy=AgeDown")
        keyMap.putI("ikey", 5)
        keyMap.putS("skey", -3)
        keyMap.putB("bkey", 2)
        keyMap.putD("dkey", 3.14)
        keyMap.putF("fkey", 2.78)

        self.assertEqual(len(keyMap), 5)

        self.assertEqual(keyMap.key(0), "ikey")
        self.assertEqual(keyMap.key(1), "skey")
        self.assertEqual(keyMap.key(2), "bkey")
        self.assertEqual(keyMap.key(3), "dkey")
        self.assertEqual(keyMap.key(4), "fkey")

    def test_KeyMapScalars(self):
        keyMap = astshim.KeyMap()
        zoomMap = astshim.ZoomMap(2, 5)
        keyMap.putI("ikey", 5)
        keyMap.putS("skey", -3)
        keyMap.putB("bkey", 2)
        keyMap.putD("dkey", 3.14)
        keyMap.putF("fkey", 2.78)
        keyMap.putC("ckey", "strvalue")
        keyMap.putA("akey", zoomMap)

        self.assertEqual(len(keyMap), 7)
        self.assertEqual(keyMap.length("ikey"), 1)
        self.assertEqual(keyMap.length("skey"), 1)
        self.assertEqual(keyMap.length("bkey"), 1)
        self.assertEqual(keyMap.length("dkey"), 1)
        self.assertEqual(keyMap.length("fkey"), 1)
        self.assertEqual(keyMap.length("ckey"), 1)
        self.assertEqual(keyMap.length("akey"), 1)

        self.assertEqual(keyMap.type("ikey"), astshim.DataType.IntType)
        self.assertEqual(keyMap.type("skey"), astshim.DataType.ShortIntType)
        self.assertEqual(keyMap.type("bkey"), astshim.DataType.ByteType)
        self.assertEqual(keyMap.type("dkey"), astshim.DataType.DoubleType)
        self.assertEqual(keyMap.type("fkey"), astshim.DataType.FloatType)
        self.assertEqual(keyMap.type("ckey"), astshim.DataType.StringType)
        self.assertEqual(keyMap.type("akey"), astshim.DataType.ObjectType)
        self.assertEqual(keyMap.type("no"), astshim.DataType.BadType)

        self.assertEqual(keyMap.getI("ikey"), [5])
        self.assertEqual(keyMap.getI("ikey", 0), 5)
        self.assertEqual(keyMap.getS("skey"), [-3])
        self.assertEqual(keyMap.getS("skey", 0), -3)
        self.assertEqual(keyMap.getB("bkey"), [2])
        self.assertEqual(keyMap.getB("bkey", 0), 2)
        assert_allclose(keyMap.getD("dkey"), [3.14])
        self.assertAlmostEqual(keyMap.getD("dkey", 0), 3.14)
        assert_allclose(keyMap.getF("fkey"), [2.78])
        self.assertAlmostEqual(keyMap.getF("fkey", 0), 2.78)
        self.assertEqual(keyMap.getC("ckey"), ["strvalue"])
        self.assertEqual(keyMap.getC("ckey", 0), "strvalue")
        self.assertEqual([obj.show() for obj in keyMap.getA("akey")], [zoomMap.show()])
        self.assertEqual(keyMap.getA("akey", 0).show(), zoomMap.show())

        self.assertEqual(keyMap.getC("CkEy"), [])  # invalid key (case is wrong)
        with self.assertRaises(Exception):
            keyMap.getC("CKey", 0)  # invalid key (case is wrong)
        with self.assertRaises(Exception):
            keyMap.getC("ckey", 1)  # invalid index

    def test_KeyMapCaseBlind(self):
        keyMap = astshim.KeyMap("KeyCase=0")
        keyMap.putC("ckey", "strvalue")

        self.assertEqual(keyMap.getC("CKey"), ["strvalue"])
        self.assertEqual(keyMap.getC("CKey", 0), "strvalue")

    def test_KeyMapRename(self):
        keyMap = astshim.KeyMap()
        keyMap.putC("ckey", "strvalue")
        keyMap.rename("ckey", "new")
        self.assertEqual(len(keyMap), 1)
        self.assertEqual(keyMap.getC("ckey"), [])
        self.assertEqual(keyMap.getC("new"), ["strvalue"])

    def test_KeyMapRemove(self):
        keyMap = astshim.KeyMap()
        keyMap.putC("ckey", "strvalue")
        keyMap.remove("ckey")
        self.assertEqual(len(keyMap), 0)
        self.assertEqual(keyMap.getC("ckey"), [])

    def test_KeyMapDefinedHasKey(self):
        keyMap = astshim.KeyMap()
        keyMap.putC("ckey", "strvalue")
        keyMap.putU("ukey")

        self.assertTrue(keyMap.hasKey("ckey"))
        self.assertTrue(keyMap.defined("ckey"))
        self.assertTrue(keyMap.hasKey("ukey"))
        self.assertFalse(keyMap.defined("ukey"))
        self.assertFalse(keyMap.hasKey("no"))
        self.assertFalse(keyMap.defined("no"))

    def testKeyMapKeys(self):
        keyMap = astshim.KeyMap("SortBy=AgeDown")
        keyMap.putI("ikey", 5)
        keyMap.putS("skey", -3)
        keyMap.putB("bkey", 2)
        keyMap.putD("dkey", 3.14)
        keyMap.putF("fkey", 2.78)
        keyMap.putC("ckey", "strvalue")

        desKeys = ["ikey", "skey", "bkey", "dkey", "fkey", "ckey"]

        for i, key in enumerate(keyMap.keys()):
            self.assertEqual(key, desKeys[i])

    def test_KeyMapVectors(self):
        keyMap = astshim.KeyMap()
        zoomMap = astshim.ZoomMap(2, 5)
        shiftMap = astshim.ShiftMap([3.3, -4.1])
        keyMap.putI("ikey", [5, 2])
        keyMap.putS("skey", [-3, -1])
        keyMap.putB("bkey", [0, 2, 4, 8])
        keyMap.putD("dkey", [3.14, 0.005, 9.123e5])
        keyMap.putF("fkey", [2.78, 999.9])
        keyMap.putC("ckey", ["val0", "val1", "a longer value"])
        keyMap.putA("akey", [zoomMap, shiftMap])

        self.assertEqual(len(keyMap), 7)
        self.assertEqual(keyMap.length("ikey"), 2)
        self.assertEqual(keyMap.length("skey"), 2)
        self.assertEqual(keyMap.length("bkey"), 4)
        self.assertEqual(keyMap.length("dkey"), 3)
        self.assertEqual(keyMap.length("fkey"), 2)
        self.assertEqual(keyMap.length("ckey"), 3)
        self.assertEqual(keyMap.length("akey"), 2)

        self.assertEqual(keyMap.type("ikey"), astshim.DataType.IntType)
        self.assertEqual(keyMap.type("skey"), astshim.DataType.ShortIntType)
        self.assertEqual(keyMap.type("bkey"), astshim.DataType.ByteType)
        self.assertEqual(keyMap.type("dkey"), astshim.DataType.DoubleType)
        self.assertEqual(keyMap.type("fkey"), astshim.DataType.FloatType)
        self.assertEqual(keyMap.type("ckey"), astshim.DataType.StringType)
        self.assertEqual(keyMap.type("akey"), astshim.DataType.ObjectType)
        self.assertEqual(keyMap.type("no"), astshim.DataType.BadType)

        self.assertEqual(keyMap.getI("ikey"), [5, 2])
        self.assertEqual(keyMap.getS("skey"), [-3, -1])
        self.assertEqual(keyMap.getB("bkey"), [0, 2, 4, 8])
        self.assertEqual(keyMap.getB("bkey", 0), 0)
        self.assertEqual(keyMap.getB("bkey", 3), 8)
        assert_allclose(keyMap.getD("dkey"), [3.14, 0.005, 9.123e5])
        assert_allclose(keyMap.getF("fkey"), [2.78, 999.9])
        self.assertEqual(keyMap.getC("ckey"), ["val0", "val1", "a longer value"])
        self.assertEqual([obj.show() for obj in keyMap.getA("akey")], [zoomMap.show(), shiftMap.show()])

        for i, val in enumerate(keyMap.getI("ikey")):
            self.assertEqual(keyMap.getI("ikey", i), val)
        for i, val in enumerate(keyMap.getS("skey")):
            self.assertEqual(keyMap.getS("skey", i), val)
        for i, val in enumerate(keyMap.getB("bkey")):
            self.assertEqual(keyMap.getB("bkey", i), val)
        for i, val in enumerate(keyMap.getD("dkey")):
            self.assertAlmostEqual(keyMap.getD("dkey", i), val)
        for i, val in enumerate(keyMap.getF("fkey")):
            self.assertAlmostEqual(keyMap.getF("fkey", i), val, places=5)
        for i, val in enumerate(keyMap.getC("ckey")):
            self.assertEqual(keyMap.getC("ckey", i), val)
        for i, val in enumerate(keyMap.getA("akey")):
            self.assertEqual(keyMap.getA("akey", i).show(), val.show())

        clen = keyMap.length("ckey")
        with self.assertRaises(Exception):
            keyMap.getC("ckey", clen)  # invalid index
        with self.assertRaises(Exception):
            keyMap.replace("ckey", clen, "anything")  # invalid index

        keyMap.replace("ikey", 1, 3)
        self.assertEqual(keyMap.getI("ikey"), [5, 3])
        keyMap.replace("ikey", 0, -3)
        self.assertEqual(keyMap.getI("ikey"), [-3, 3])
        keyMap.append("ikey", 5)
        self.assertEqual(keyMap.getI("ikey"), [-3, 3, 5])

        keyMap.replace("skey", 1, 2)
        self.assertEqual(keyMap.getS("skey"), [-3, 2])
        keyMap.replace("skey", 0, 47)
        self.assertEqual(keyMap.getS("skey"), [47, 2])
        keyMap.append("skey", -35)
        self.assertEqual(keyMap.getS("skey"), [47, 2, -35])

        keyMap.replace("bkey", 0, 36)
        self.assertEqual(keyMap.getB("bkey"), [36, 2, 4, 8])
        keyMap.replace("bkey", 2, 0)
        self.assertEqual(keyMap.getB("bkey"), [36, 2, 0, 8])
        keyMap.replace("bkey", 1, 11)
        self.assertEqual(keyMap.getB("bkey"), [36, 11, 0, 8])
        keyMap.replace("bkey", 3, 77)
        self.assertEqual(keyMap.getB("bkey"), [36, 11, 0, 77])
        keyMap.append("bkey", 2)
        self.assertEqual(keyMap.getB("bkey"), [36, 11, 0, 77, 2])

        keyMap.replace("dkey", 1, 33.3)
        assert_allclose(keyMap.getD("dkey"), [3.14, 33.3, 9.123e5])
        keyMap.replace("dkey", 2, 152)
        assert_allclose(keyMap.getD("dkey"), [3.14, 33.3, 152])
        keyMap.replace("dkey", 0, 0.01)
        assert_allclose(keyMap.getD("dkey"), [0.01, 33.3, 152])

        keyMap.replace("fkey", 1, 3.012)
        assert_allclose(keyMap.getF("fkey"), [2.78, 3.012])
        keyMap.replace("fkey", 0, -32.1)
        assert_allclose(keyMap.getF("fkey"), [-32.1, 3.012])
        keyMap.append("fkey", 98.6)
        assert_allclose(keyMap.getF("fkey"), [-32.1, 3.012, 98.6])


if __name__ == "__main__":
    unittest.main()
