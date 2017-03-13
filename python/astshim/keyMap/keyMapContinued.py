from __future__ import absolute_import, division, print_function

from .keyMap import KeyMap

__all__ = []  # import only for side effects


def keys(self):
    for i in range(len(self)):
        yield self.key(i)
KeyMap.keys = keys
