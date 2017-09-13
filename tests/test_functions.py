from __future__ import absolute_import, division, print_function
import unittest

import astshim as ast


class TestFunctions(unittest.TestCase):

    """Test free functions"""

    def test_escape(self):
        self.assertFalse(ast.escapes())
        self.assertFalse(ast.escapes(1))
        self.assertTrue(ast.escapes(-1))
        self.assertTrue(ast.escapes(0))
        self.assertFalse(ast.escapes())


if __name__ == "__main__":
    unittest.main()
