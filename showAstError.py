import os
import multiprocessing
import astshim as ast
from astshim.detail import watchPointer


if False:  # set True to watch pointers
    watchPointer(0x5f101)


class PickleableUnitMap(ast.UnitMap):
    """The simplest possible pickle support"""
    def __init__(self, nin):
        print("Make a PickleableUnitMap in PID =", os.getpid())
        ast.UnitMap.__init__(self, nin)

    def __reduce__(self):
        print("Pickle a PickleableUnitMap in PID =", os.getpid())
        return (PickleableUnitMap, (self.nIn,))


if __name__ == "__main__":
    print("Main PID =", os.getpid())
    numProcesses = 1
    params = [1]*numProcesses
    pool = multiprocessing.Pool()
    pool.map(PickleableUnitMap, params)
    pool.close()
    pool.join()
