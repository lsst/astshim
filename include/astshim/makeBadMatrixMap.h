/* 
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef ASTSHIM_MAKEBADMATRIXMAP_H
#define ASTSHIM_MAKEBADMATRIXMAP_H

#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
Create a MatrixMap from a 2-d matrix

@warning THIS FACTORY FUNCTION IS ONLY FOR TESTING.
Use the class @ref MatrixMap for real work.
The reason this function exists is to diagnose a memory issue.
The unit test test_matrixMath exhibits an issue with this function on my Mac
that suggests some kind of memory corruption.

@param[in] matrix  The transformation matrix, where:
                    - The number of input coordinates is the number of columns.
                    - The number of output coordinates is the number of rows.
*/
Mapping makeBadMatrixMap(ndarray::Array<double, 2, 2> const & matrix) {
    const int nin = matrix.getSize<1>();
    const int nout = matrix.getSize<0>();
    int const form = 0;  // 0 = full matrix, 1 = diagonal elements only
    AstMatrixMap *map = astMatrixMap(nin, nout, form, matrix.getData(), "");
    return Mapping(reinterpret_cast<AstMapping *>(map));
}

/**
Variant factory function that simply raises an exception.

Its presence is necessary to expose the bug described in the
matrix version of makeBadMatrixMap above.
*/
Mapping makeBadMatrixMap(std::vector<double> const & diag) {
    throw std::runtime_error("Use the class MatrixMap instead");
}

}  // namespace ast

#endif
