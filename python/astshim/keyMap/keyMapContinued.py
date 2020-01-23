__all__ = ["KeyMap"]

from .keyMap import KeyMap


def keys(self):
    for i in range(len(self)):
        yield self.key(i)


KeyMap.keys = keys
