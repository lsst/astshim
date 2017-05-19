from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

import astshim
from astshim.test import ObjectTestCase


class TestObject(ObjectTestCase):

    def test_attributes(self):
        """Test accessing object attributes
        """
        nin = 2
        zoom = 1.3
        obj = astshim.ZoomMap(nin, zoom)

        self.assertEqual(obj.getClass(), "ZoomMap")

        self.assertTrue(obj.hasAttribute("ID"))
        self.assertTrue(obj.hasAttribute("Ident"))
        self.assertTrue(obj.hasAttribute("UseDefs"))

        self.assertEqual(obj.getID(), "")
        self.assertEqual(obj.getIdent(), "")
        self.assertEqual(obj.getUseDefs(), True)

    def test_clear_and_test(self):
        """Test Object.clear and Object.test"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertFalse(obj.test("ID"))
        obj.setID("initial_id")
        self.assertEqual(obj.getID(), "initial_id")
        self.assertTrue(obj.test("ID"))
        obj.clear("ID")
        self.assertEqual(obj.getID(), "")
        self.assertFalse(obj.test("ID"))

    def test_copy_and_same(self):
        """Test Object.copy and Object.same"""
        obj = astshim.ZoomMap(2, 1.3, "Ident=original")

        # there may be more than one object in existence if run with pytest
        initialNumObj = obj.getNObject()

        self.checkCopy(obj)
        cp = obj.copy()
        # a deep copy does not increment
        self.assertEqual(obj.getRefCount(), 1)

        seriesMap = obj.of(obj)
        # obj itself plus two copies in the SeriesMap
        self.assertEqual(obj.getRefCount(), 3)
        del seriesMap
        self.assertEqual(obj.getRefCount(), 1)

        cp.setIdent("copy")
        self.assertEqual(cp.getIdent(), "copy")
        self.assertEqual(obj.getIdent(), "original")

        del cp
        self.assertEqual(obj.getNObject(), initialNumObj)
        self.assertEqual(obj.getRefCount(), 1)

    def test_error_handling(self):
        """Test handling of AST errors
        """
        # To introduce an AST error construct a PolyMap with no inverse mapping
        # and then try to transform in the inverse direction.
        coeff_f = np.array([
            [1.2, 1, 2, 0],
            [-0.5, 1, 1, 1],
            [1.0, 2, 0, 1],
        ])
        pm = astshim.PolyMap(coeff_f, 2, "IterInverse=0")
        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        # make sure the error string contains "Error"
        try:
            pm.tranInverse(indata)
        except RuntimeError as e:
            self.assertEqual(e.args[0].count("Error"), 1)
            print(e)

        # cause another error and make sure the first error message was purged
        try:
            pm.tranInverse(indata)
        except RuntimeError as e:
            self.assertEqual(e.args[0].count("Error"), 1)

    def test_id(self):
        """Test that ID is *not* transferred to copies"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertEqual(obj.getID(), "")
        obj.setID("initial_id")
        self.assertEqual(obj.getID(), "initial_id")
        cp = obj.copy()
        self.assertEqual(cp.getID(), "")

    def test_ident(self):
        """Test that Ident *is* transferred to copies"""
        obj = astshim.ZoomMap(2, 1.3)

        self.assertEqual(obj.getIdent(), "")
        obj.setIdent("initial_ident")
        self.assertEqual(obj.getIdent(), "initial_ident")
        cp = obj.copy()
        self.assertEqual(cp.getIdent(), "initial_ident")


if __name__ == "__main__":
    unittest.main()
