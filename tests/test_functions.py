from __future__ import absolute_import, division, print_function
import unittest

import astshim


class TestFunctions(unittest.TestCase):

    """Test free functions"""

    def test_escape(self):
        self.assertFalse(astshim.escapes())
        self.assertFalse(astshim.escapes(1))
        self.assertTrue(astshim.escapes(-1))
        self.assertTrue(astshim.escapes(0))
        self.assertFalse(astshim.escapes())


if __name__ == "__main__":
    unittest.main()
