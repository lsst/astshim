from __future__ import absolute_import, division, print_function
import unittest

import astshim
from astshim.test import ObjectTestCase


class TestObject(ObjectTestCase):

    def test_attributes(self):
        """Test accessing object attributes
        """
        nin = 2
        zoom = 1.3
        obj = astshim.ZoomMap(nin, zoom)

        self.assertEquals(obj.getClass(), "ZoomMap")

        self.assertTrue(obj.hasAttribute("ID"))
        self.assertTrue(obj.hasAttribute("Ident"))
        self.assertTrue(obj.hasAttribute("UseDefs"))

        self.assertEquals(obj.getID(), "")
        self.assertEquals(obj.getIdent(), "")
        self.assertEquals(obj.getUseDefs(), True)

    def test_unknown_attributes(self):
        """Test accessing unknown attributes"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertFalse(obj.hasAttribute("NonExistentAttribute"))

        with self.assertRaises(Exception):
            obj.getC("NonExistentAttribute")
        with self.assertRaises(Exception):
            obj.getF("NonExistentAttribute")
        with self.assertRaises(Exception):
            obj.getD("NonExistentAttribute")
        with self.assertRaises(Exception):
            obj.getI("NonExistentAttribute")
        with self.assertRaises(Exception):
            obj.getL("NonExistentAttribute")

    def test_clear_and_test(self):
        """Test Object.clear and Object.test"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertFalse(obj.test("ID"))
        obj.setID("initial_id")
        self.assertEquals(obj.getID(), "initial_id")
        self.assertTrue(obj.test("ID"))
        obj.clear("ID")
        self.assertEquals(obj.getID(), "")
        self.assertFalse(obj.test("ID"))

    def test_copy_and_same(self):
        """Test Object.copy and Object.same"""
        obj = astshim.ZoomMap(2, 1.3, "Ident=original")
        self.checkCopy(obj)
        cp = obj.copy()

        obj2 = obj.of(obj)
        self.assertEqual(obj.getRefCount(), 3)  # obj itself plus two copies in the CmpMap
        del obj2
        self.assertEqual(obj.getRefCount(), 1)

        cp.setIdent("copy")
        self.assertEqual(cp.getIdent(), "copy")
        self.assertEqual(obj.getIdent(), "original")

        del cp
        self.assertEqual(obj.getNobject(), 1)
        self.assertEqual(obj.getRefCount(), 1)

    def test_id(self):
        """Test that ID is *not* transferred to copies"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertEquals(obj.getID(), "")
        obj.setID("initial_id")
        self.assertEquals(obj.getID(), "initial_id")
        cp = obj.copy()
        self.assertEquals(cp.getID(), "")

    def test_ident(self):
        """Test that Ident *is* transferred to copies"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertEquals(obj.getIdent(), "")
        obj.setIdent("initial_ident")
        self.assertEquals(obj.getIdent(), "initial_ident")
        cp = obj.copy()
        self.assertEquals(cp.getIdent(), "initial_ident")


if __name__ == "__main__":
    unittest.main()
