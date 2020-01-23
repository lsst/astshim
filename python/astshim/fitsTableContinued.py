__all__ = ["FitsTable", "getColumnData"]

from . import base
from .fitsTable import FitsTable


def getColumnData(self, column):
    """Retrieve the column data in the correct type and shape.

    Parameters
    ----------
    column : `str`
        Name of the column to retrieve.

    Returns
    -------
    data : `list` of `numpy.array`

    """
    nrows = self.nRow
    shape = self.columnShape(column)
    dtype = self.columnType(column)

    if dtype == base.DataType.DoubleType:
        newshape = list(shape)
        newshape.append(nrows)
        coldata = self.getColumnData1D(column)
        coldata = coldata.reshape(newshape, order="F")
    else:
        raise ValueError("Can only retrieve double column data")
    return coldata


FitsTable.getColumnData = getColumnData
