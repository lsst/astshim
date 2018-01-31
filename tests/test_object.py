from __future__ import absolute_import, division, print_function
import multiprocessing
import unittest

import numpy as np

import astshim as ast
from astshim.test import ObjectTestCase


class PickleableUnitMap(ast.UnitMap):
    def __reduce__(self):
        return (PickleableUnitMap, (self.nIn,))


class TestObject(ObjectTestCase):

    def test_attributes(self):
        """Test accessing object attributes
        """
        nin = 2
        zoom = 1.3
        obj = ast.ZoomMap(nin, zoom)

        self.assertEqual(obj.className, "ZoomMap")

        self.assertTrue(obj.hasAttribute("ID"))
        self.assertTrue(obj.hasAttribute("Ident"))
        self.assertTrue(obj.hasAttribute("UseDefs"))

        self.assertEqual(obj.id, "")
        self.assertEqual(obj.ident, "")
        self.assertEqual(obj.useDefs, True)

    def test_clear_and_test(self):
        """Test Object.clear and Object.test"""
        obj = ast.ZoomMap(2, 1.3)

        self.assertFalse(obj.test("ID"))
        obj.id = "initial_id"
        self.assertEqual(obj.id, "initial_id")
        self.assertTrue(obj.test("ID"))
        obj.clear("ID")
        self.assertEqual(obj.id, "")
        self.assertFalse(obj.test("ID"))

    def test_copy_and_same(self):
        """Test Object.copy and Object.same"""
        obj = ast.ZoomMap(2, 1.3, "Ident=original")

        initialNumObj = obj.getNObject()  # may be >1 when run using pytest

        self.checkCopy(obj)
        cp = obj.copy()
        # A deep copy does not increment refCount but does incremente nObject
        self.assertEqual(obj.getRefCount(), 1)
        self.assertEqual(obj.getNObject(), initialNumObj + 1)
        # A deep copy is not the `same` as the original:
        # `same` compares AST pointers, similar to Python `is`
        self.assertFalse(obj.same(cp))
        self.assertTrue(obj.same(obj))

        cp.ident = "copy"
        self.assertEqual(cp.ident, "copy")
        self.assertEqual(obj.ident, "original")

        del cp
        self.assertEqual(obj.getNObject(), initialNumObj)
        self.assertEqual(obj.getRefCount(), 1)

        seriesMap = obj.then(obj)
        # The seriesMap contains two shallow copies of `obj`, so refCount
        # is increased by 2 and nObject remains unchanged
        self.assertEqual(obj.getRefCount(), 3)
        self.assertEqual(obj.getNObject(), initialNumObj)
        del seriesMap
        self.assertEqual(obj.getRefCount(), 1)
        self.assertEqual(obj.getNObject(), initialNumObj)

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
        pm = ast.PolyMap(coeff_f, 2, "IterInverse=0")
        indata = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ])

        # make sure the error string contains "Error"
        try:
            pm.applyInverse(indata)
        except RuntimeError as e:
            self.assertEqual(e.args[0].count("Error"), 1)
            print(e)

        # cause another error and make sure the first error message was purged
        try:
            pm.applyInverse(indata)
        except RuntimeError as e:
            self.assertEqual(e.args[0].count("Error"), 1)

    def test_equality(self):
        """Test __eq__ and __ne__
        """
        frame = ast.Frame(2)
        zoomMap = ast.ZoomMap(2, 1.5)
        frameSet1 = ast.FrameSet(frame, zoomMap, frame)
        frameSet2 = ast.FrameSet(frame, zoomMap, frame)
        self.assertTrue(frameSet1 == frameSet2)
        self.assertFalse(frameSet1 != frameSet2)
        self.assertEqual(frameSet1, frameSet2)

        # the base attribute of frameSet1 is not set; set the base attribute
        # of framesSet2 and make sure the frame sets are now not equal
        self.assertFalse(frameSet1.test("Base"))
        frameSet2.base = 1
        self.assertTrue(frameSet2.test("Base"))
        self.assertFalse(frameSet1 == frameSet2)
        self.assertTrue(frameSet1 != frameSet2)
        self.assertNotEqual(frameSet1, frameSet2)

        # make sure base is unset in the inverse of the inverse of frameSet1,
        # else the equality test will fail for hard-to-understand reasons
        self.assertFalse(frameSet1.getInverse().getInverse().test("Base"))
        self.assertNotEqual(frameSet1, frameSet1.getInverse())
        self.assertEqual(frameSet1, frameSet1.getInverse().getInverse())
        self.assertFalse(frameSet1.getInverse().getInverse().test("Base"))

        frame3 = ast.Frame(2)
        frame3.title = "Frame 3"
        frameSet3 = ast.FrameSet(frame3)
        self.assertNotEqual(frameSet1, frameSet3)

    def test_id(self):
        """Test that ID is *not* transferred to copies"""
        obj = ast.ZoomMap(2, 1.3)

        self.assertEqual(obj.id, "")
        obj.id = "initial_id"
        self.assertEqual(obj.id, "initial_id")
        cp = obj.copy()
        self.assertEqual(cp.id, "")

    def test_ident(self):
        """Test that Ident *is* transferred to copies"""
        obj = ast.ZoomMap(2, 1.3)

        self.assertEqual(obj.ident, "")
        obj.ident = "initial_ident"
        self.assertEqual(obj.ident, "initial_ident")
        cp = obj.copy()
        self.assertEqual(cp.ident, "initial_ident")

    def test_multiprocessing(self):
        """Make sure we can return objects from multiprocessing

        This tests DM-13316: AST errors when using multiprocessing
        to return astshim objects.
        """
        numProcesses = 2
        naxes = 1
        params = [naxes]*numProcesses
        pool = multiprocessing.Pool(processes=numProcesses)
        results = pool.map(PickleableUnitMap, params)
        self.assertEqual(results, [PickleableUnitMap(naxes)]*numProcesses)

    def test_show(self):
        # pick an object with no floats so we don't have to worry
        # about the float representation
        obj = ast.Frame(2)
        desShowLines = [
            " Begin Frame \t# Coordinate system description",
            "#   Title = \"2-d coordinate system\" \t# Title of coordinate system",
            "    Naxes = 2 \t# Number of coordinate axes",
            "#   Lbl1 = \"Axis 1\" \t# Label for axis 1",
            "#   Lbl2 = \"Axis 2\" \t# Label for axis 2",
            "    Ax1 = \t# Axis number 1",
            "       Begin Axis \t# Coordinate axis",
            "       End Axis",
            "    Ax2 = \t# Axis number 2",
            "       Begin Axis \t# Coordinate axis",
            "       End Axis",
            " End Frame",
            "",
        ]
        desShowLinesNoComments = [
            " Begin Frame",
            "#   Title = \"2-d coordinate system\"",
            "    Naxes = 2",
            "#   Lbl1 = \"Axis 1\"",
            "#   Lbl2 = \"Axis 2\"",
            "    Ax1 =",
            "       Begin Axis",
            "       End Axis",
            "    Ax2 =",
            "       Begin Axis",
            "       End Axis",
            " End Frame",
            "",
        ]
        self.assertEqual(obj.show(), "\n".join(desShowLines))
        self.assertEqual(obj.show(True), "\n".join(desShowLines))
        self.assertEqual(obj.show(False), "\n".join(desShowLinesNoComments))


if __name__ == "__main__":
    unittest.main()
