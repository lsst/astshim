import unittest

import numpy as np

from . import astshimLib as astshim


class ObjectTestCase(unittest.TestCase):

    """Base class for unit tests of objects
    """

    def checkCopy(self, obj):
        nobj = obj.getNobject()
        nref = obj.getRefCount()
        cp = obj.copy()
        self.assertEqual(type(obj), type(cp))
        self.assertEqual(str(obj), str(cp))
        self.assertEqual(repr(obj), repr(cp))
        self.assertEqual(obj.getNobject(), nobj + 1)
        # Object.copy makes a new pointer instead of copying the old one,
        # so the reference count of the old one does not increase
        self.assertEqual(obj.getRefCount(), nref)
        self.assertFalse(obj.same(cp))
        self.assertEqual(cp.getNobject(), nobj + 1)
        self.assertEqual(cp.getRefCount(), 1)

    def checkPersistence(self, obj):
        """Check that an Ast object can be persisted and unpersisted
        """
        # round trip with a Channel
        ss1 = astshim.StringStream()
        chan1 = astshim.Channel(ss1)
        chan1.write(obj)
        ss1.sinkToSource()
        obj_copy1 = chan1.read()
        self.assertEqual(obj.getClass(), obj_copy1.getClass())
        self.assertEqual(obj.show(), obj_copy1.show())
        self.assertEqual(str(obj), str(obj_copy1))
        self.assertEqual(repr(obj), repr(obj_copy1))

        # round trip with an XmlChan
        ss2 = astshim.StringStream()
        chan2 = astshim.XmlChan(ss2)
        chan2.write(obj)
        ss2.sinkToSource()
        obj_copy2 = chan2.read()
        self.assertEqual(obj.getClass(), obj_copy2.getClass())
        self.assertEqual(obj.show(), obj_copy2.show())
        self.assertEqual(str(obj), str(obj_copy2))
        self.assertEqual(repr(obj), repr(obj_copy2))


class MappingTestCase(ObjectTestCase):

    """Base class for unit tests of mappings
    """

    def checkRoundTrip(self, amap, poslist, rtol=1e-05, atol=1e-08):
        """Check that a mapping's reverse transform is the opposite of forward

        amap is the mapping to test
        poslist is a list of input position for a forward transform;
            a numpy array with shape [nin, num points]
            or collection that can be cast to same
        rtol is the relative tolerance for np.allclose
        atol is the absolute tolerance for np.allclose
        """
        poslist = np.array(poslist, dtype=float)
        # forward with tran, inverse with tranInverse
        to_poslist = amap.tran(poslist)
        rt_poslist = amap.tranInverse(to_poslist)
        self.assertTrue(np.allclose(poslist, rt_poslist))

        # forward with tran, inverse with getInverse().tran
        amapinv = amap.getInverse()
        rt2_poslist = amapinv.tran(to_poslist)
        self.assertTrue(np.allclose(poslist, rt2_poslist))

        # forward and inverse with a compound map of amap.getInverse().of(amap)
        acmp = amapinv.of(amap)
        self.assertTrue(np.allclose(poslist, acmp.tran(poslist)))

    def checkBasicSimplify(self, amap):
        """Check basic simplfication for a reversible mapping

        Check the following:
        - A compound mapping of a amap and its inverse simplifies to a unit amap
        - A compound mapping of a amap and a unit amap simplifies to the original amap
        """
        amapinv = amap.getInverse()
        cmp1 = amapinv.of(amap)
        unit1 = cmp1.simplify()
        self.assertEqual(unit1.getClass(), "UnitMap")
        self.assertEqual(amap.getNin(), cmp1.getNin())
        self.assertEqual(amap.getNin(), cmp1.getNout())
        self.assertEqual(cmp1.getNin(), unit1.getNin())
        self.assertEqual(cmp1.getNout(), unit1.getNout())

        cmp2 = amap.of(amapinv)
        unit2 = cmp2.simplify()
        self.assertEqual(unit2.getClass(), "UnitMap")
        self.assertEqual(amapinv.getNin(), cmp2.getNin())
        self.assertEqual(amapinv.getNin(), cmp2.getNout())
        self.assertEqual(cmp2.getNin(), unit2.getNin())
        self.assertEqual(cmp2.getNout(), unit2.getNout())

        for ma, mb, desmap3 in (
            (unit1, amap, amap),
            (amap, unit2, amap),
            (unit2, amapinv, amapinv),
            (amapinv, unit1, amapinv),
        ):
            cmp3 = mb.of(ma)
            cmp3simp = cmp3.simplify()
            self.assertEqual(cmp3simp.getClass(), amap.simplify().getClass())
            self.assertEqual(ma.getNin(), cmp3.getNin())
            self.assertEqual(mb.getNout(), cmp3.getNout())
            self.assertEqual(cmp3.getNin(), cmp3simp.getNin())
            self.assertEqual(cmp3.getNout(), cmp3simp.getNout())
