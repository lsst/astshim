from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_equal

import astshim as ast
from astshim.test import ObjectTestCase


class TestBase(ObjectTestCase):

    def test_arrayFromVector(self):
        nAxes = 3
        nValues = 5
        np.random.seed(1)
        dataVec = np.random.rand(nAxes * nValues)
        desiredDataArr = dataVec.copy()
        desiredDataArr.shape = (nAxes, nValues)
        dataArr = ast.arrayFromVector(vec=dataVec, nAxes=nAxes)
        assert_equal(dataArr, desiredDataArr)

        dataArr2 = ast.arrayFromVector(vec=list(dataVec), nAxes=nAxes)
        assert_equal(dataArr2, desiredDataArr)

        # make sure dataArr is a deep copy; changing dataVec should not change dataArr
        dataVec[0] += 10
        assert_equal(dataArr, desiredDataArr)

        for delta in (-1, 1):
            badDataVec = np.random.rand(nAxes * nValues + delta)
            with self.assertRaises(RuntimeError):
                ast.arrayFromVector(vec=badDataVec, nAxes=nAxes)


if __name__ == "__main__":
    unittest.main()
