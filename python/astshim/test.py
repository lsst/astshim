import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .channel import Channel
from .fitsChan import FitsChan
from .polyMap import PolyMap
from .xmlChan import XmlChan
from .stream import StringStream


class ObjectTestCase(unittest.TestCase):

    """Base class for unit tests of objects
    """

    def checkCopy(self, obj):
        """Check that an astshim object can be deep-copied
        """
        nobj = obj.getNObject()
        nref = obj.getRefCount()
        cp = obj.copy()
        self.assertEqual(type(obj), type(cp))
        self.assertEqual(str(obj), str(cp))
        self.assertEqual(repr(obj), repr(cp))
        self.assertEqual(obj.getNObject(), nobj + 1)
        # Object.copy makes a new pointer instead of copying the old one,
        # so the reference count of the old one does not increase
        self.assertEqual(obj.getRefCount(), nref)
        self.assertFalse(obj.same(cp))
        self.assertEqual(cp.getNObject(), nobj + 1)
        self.assertEqual(cp.getRefCount(), 1)

        del cp
        self.assertEqual(obj.getNObject(), nobj)
        self.assertEqual(obj.getRefCount(), nref)

    def checkPersistence(self, obj):
        """Check that an astshim object can be persisted and unpersisted

        Check persistence using Channel, FitsChan (with native encoding,
        as the only encoding compatible with all AST objects),
        and XmlChan
        """
        for channelType, options in (
            (Channel, ""),
            (FitsChan, "Encoding=Native"),
            (XmlChan, ""),
        ):
            ss = StringStream()
            chan = Channel(ss)
            chan.write(obj)
            ss.sinkToSource()
            obj_copy = chan.read()
            self.assertEqual(obj.className, obj_copy.className)
            self.assertEqual(obj.show(), obj_copy.show())
            self.assertEqual(str(obj), str(obj_copy))
            self.assertEqual(repr(obj), repr(obj_copy))


class MappingTestCase(ObjectTestCase):

    """Base class for unit tests of mappings
    """

    def checkRoundTrip(self, amap, poslist, rtol=1e-05, atol=1e-08):
        """Check that a mapping's reverse transform is the opposite of forward

        amap is the mapping to test
        poslist is a list of input position for a forward transform;
            a numpy array with shape [nin, num points]
            or collection that can be cast to same
        rtol is the relative tolerance for numpy.testing.assert_allclose
        atol is the absolute tolerance for numpy.testing.assert_allclose
        """
        poslist = np.array(poslist, dtype=float)
        if len(poslist.shape) == 1:
            # supplied data was a single list of points
            poslist.shape = (1, len(poslist))
        # forward with applyForward, inverse with applyInverse
        to_poslist = amap.applyForward(poslist)
        rt_poslist = amap.applyInverse(to_poslist)
        assert_allclose(poslist, rt_poslist, rtol=rtol, atol=atol)

        # forward with applyForward, inverse with getInverse().applyForward
        amapinv = amap.getInverse()
        rt2_poslist = amapinv.applyForward(to_poslist)
        assert_allclose(poslist, rt2_poslist, rtol=rtol, atol=atol)

        # forward and inverse with a compound map of amap.then(amap.getInverse())
        acmp = amap.then(amapinv)
        assert_allclose(poslist, acmp.applyForward(poslist), rtol=rtol, atol=atol)

        # test vector versions of forward and inverse
        posvec = list(poslist.flat)
        to_posvec = amap.applyForward(posvec)
        # cast to_poslist to np.array because if poslist has 1 axis then
        # a list is returned, which has no `flat` attribute
        assert_allclose(to_posvec, list(to_poslist.flat), rtol=rtol, atol=atol)

        rt_posvec = amap.applyInverse(to_posvec)
        assert_allclose(posvec, rt_posvec, rtol=rtol, atol=atol)

    def checkBasicSimplify(self, amap):
        """Check basic simplfication for a reversible mapping

        Check the following:
        - A compound mapping of a amap and its inverse simplifies to a unit amap
        - A compound mapping of a amap and a unit amap simplifies to the original amap
        """
        amapinv = amap.getInverse()
        cmp1 = amap.then(amapinv)
        unit1 = cmp1.simplify()
        self.assertEqual(unit1.className, "UnitMap")
        self.assertEqual(amap.nIn, cmp1.nIn)
        self.assertEqual(amap.nIn, cmp1.nOut)
        self.assertEqual(cmp1.nIn, unit1.nIn)
        self.assertEqual(cmp1.nOut, unit1.nOut)

        cmp2 = amapinv.then(amap)
        unit2 = cmp2.simplify()
        self.assertEqual(unit2.className, "UnitMap")
        self.assertEqual(amapinv.nIn, cmp2.nIn)
        self.assertEqual(amapinv.nIn, cmp2.nOut)
        self.assertEqual(cmp2.nIn, unit2.nIn)
        self.assertEqual(cmp2.nOut, unit2.nOut)

        for ma, mb, desmap3 in (
            (unit1, amap, amap),
            (amap, unit2, amap),
            (unit2, amapinv, amapinv),
            (amapinv, unit1, amapinv),
        ):
            cmp3 = ma.then(mb)
            cmp3simp = cmp3.simplify()
            self.assertEqual(cmp3simp.className, amap.simplify().className)
            self.assertEqual(ma.nIn, cmp3.nIn)
            self.assertEqual(mb.nOut, cmp3.nOut)
            self.assertEqual(cmp3.nIn, cmp3simp.nIn)
            self.assertEqual(cmp3.nOut, cmp3simp.nOut)

    def checkMappingPersistence(self, amap, poslist):
        """Check that a mapping gives identical answers to unpersisted copy

        poslist is a list of input position for a forward transform
            (if it exists), or the inverse transform (if not).
            A numpy array with shape [nAxes, num points]
            or collection that can be cast to same

        Checks each direction, if present. However, for generality,
        does not check that the two directions are inverses of each other;
        call checkRoundTrip for that.

        Does everything checkPersistence does, so no need to call both.
        """
        for channelType, options in (
            (Channel, ""),
            (FitsChan, "Encoding=Native"),
            (XmlChan, ""),
        ):
            ss = StringStream()
            chan = Channel(ss)
            chan.write(amap)
            ss.sinkToSource()
            amap_copy = chan.read()
            self.assertEqual(amap.className, amap_copy.className)
            self.assertEqual(amap.show(), amap_copy.show())
            self.assertEqual(str(amap), str(amap_copy))
            self.assertEqual(repr(amap), repr(amap_copy))

            if amap.hasForward:
                outPoslist = amap.applyForward(poslist)
                assert_array_equal(outPoslist, amap_copy.applyForward(poslist))

                if amap.hasInverse:
                    assert_array_equal(amap.applyInverse(outPoslist),
                                       amap_copy.applyInverse(outPoslist))

            elif amap.hasInverse:
                assert_array_equal(amap.applyInverse(poslist),
                                   amap_copy.applyInverse(poslist))

            else:
                raise RuntimeError("mapping has neither forward nor inverse transform")

    def checkMemoryForCompoundObject(self, obj1, obj2, cmpObj, isSeries):
        """Check the memory usage for a compoundObject

        obj1: first object in compound object
        obj2: second object in compound object
        cmpObj: compound object (SeriesMap, ParallelMap, CmpMap or CmpFrame)
        isSeries: is compound object in series? None to not test (e.g. CmpFrame)
        """
        # if obj1 and obj2 are the same type then copying the compound object
        # will increase the NObject of each by 2, otherwise 1
        deltaObj = 2 if type(obj1) == type(obj2) else 1

        initialNumObj1 = obj1.getNObject()
        initialNumObj2 = obj2.getNObject()
        initialNumCmpObj = cmpObj.getNObject()
        initialRefCountObj1 = obj1.getRefCount()
        initialRefCountObj2 = obj2.getRefCount()
        initialRefCountCmpObj = cmpObj.getRefCount()
        self.assertEqual(obj1.getNObject(), initialNumObj1)
        self.assertEqual(obj2.getNObject(), initialNumObj2)
        if isSeries is not None:
            if isSeries is True:
                self.assertTrue(cmpObj.series)
            elif isSeries is False:
                self.assertFalse(cmpObj.series)

        # making a deep copy should increase the object count of the contained objects
        # but should not affect the reference count
        cp = cmpObj.copy()
        self.assertEqual(cmpObj.getRefCount(), initialRefCountCmpObj)
        self.assertEqual(cmpObj.getNObject(), initialNumCmpObj + 1)
        self.assertEqual(obj1.getRefCount(), initialRefCountObj1)
        self.assertEqual(obj2.getRefCount(), initialRefCountObj2)
        self.assertEqual(obj1.getNObject(), initialNumObj1 + deltaObj)
        self.assertEqual(obj2.getNObject(), initialNumObj2 + deltaObj)

        # deleting the deep copy should restore ref count and nobject
        del cp
        self.assertEqual(cmpObj.getRefCount(), initialRefCountCmpObj)
        self.assertEqual(cmpObj.getNObject(), initialNumCmpObj)
        self.assertEqual(obj1.getRefCount(), initialRefCountObj1)
        self.assertEqual(obj1.getNObject(), initialNumObj1)
        self.assertEqual(obj2.getRefCount(), initialRefCountObj2)
        self.assertEqual(obj2.getNObject(), initialNumObj2)


def makePolyMapCoeffs(nIn, nOut):
    """Make an array of coefficients for astshim.PolyMap for the following equation:

    fj(x) = C0j x0^2 + C1j x1^2 + C2j x2^2 + ... + CNj xN^2
    where:
    * i ranges from 0 to N=nIn-1
    * j ranges from 0 to nOut-1,
    * Cij = 0.001 (i+j+1)
    """
    baseCoeff = 0.001
    forwardCoeffs = []
    for out_ind in range(nOut):
        coeffOffset = baseCoeff * out_ind
        for in_ind in range(nIn):
            coeff = baseCoeff * (in_ind + 1) + coeffOffset
            coeffArr = [coeff, out_ind + 1] + [2 if i == in_ind else 0 for i in range(nIn)]
            forwardCoeffs.append(coeffArr)
    return np.array(forwardCoeffs, dtype=float)


def makeTwoWayPolyMap(nIn, nOut):
    """Make an astshim.PolyMap suitable for testing

    The forward transform is as follows:
    fj(x) = C0j x0^2 + C1j x1^2 + C2j x2^2 + ... + CNj xN^2 where Cij = 0.001 (i+j+1)

    The reverse transform is the same equation with i and j reversed
    thus it is NOT the inverse of the forward direction,
    but is something that can be easily evaluated.

    The equation is chosen for the following reasons:
    - It is well defined for any positive value of nIn, nOut
    - It stays small for small x, to avoid wraparound of angles for SpherePoint endpoints
    """
    forwardCoeffs = makePolyMapCoeffs(nIn, nOut)
    reverseCoeffs = makePolyMapCoeffs(nOut, nIn)
    polyMap = PolyMap(forwardCoeffs, reverseCoeffs)
    assert polyMap.nIn == nIn
    assert polyMap.nOut == nOut
    assert polyMap.hasForward
    assert polyMap.hasInverse
    return polyMap


def makeForwardPolyMap(nIn, nOut):
    """Make an astshim.PolyMap suitable for testing

    The forward transform is the same as for `makeTwoWayPolyMap`.
    This map does not have a reverse transform.

    The equation is chosen for the following reasons:
    - It is well defined for any positive value of nIn, nOut
    - It stays small for small x, to avoid wraparound of angles for SpherePoint endpoints
    """
    forwardCoeffs = makePolyMapCoeffs(nIn, nOut)
    polyMap = PolyMap(forwardCoeffs, nOut, "IterInverse=0")
    assert polyMap.nIn == nIn
    assert polyMap.nOut == nOut
    assert polyMap.hasForward
    assert not polyMap.hasInverse
    return polyMap
