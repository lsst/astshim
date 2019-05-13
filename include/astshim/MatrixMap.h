/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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
#ifndef ASTSHIM_MATRIXMAP_H
#define ASTSHIM_MATRIXMAP_H

#include <memory>
#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
MatrixMap is a form of @ref Mapping which performs a general linear transformation.

Each set of input coordinates, regarded as a column-vector, are pre-multiplied by a matrix
(whose elements are specified when the MatrixMap is created) to give a new column-vector
containing the output coordinates. If appropriate, the inverse transformation may also be performed.
*/
class MatrixMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a MatrixMap from a 2-d matrix

    @note The inverse transformation will only be available if `matrix` is square and non-singular.

    @param[in] matrix  The transformation matrix, where:
                        - The number of input coordinates is the number of columns.
                        - The number of output coordinates is the number of rows.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit MatrixMap(ConstArray2D const &matrix, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      // form 0 = full matrix, 1 = diagonal elements only
                      astMatrixMap(matrix.getSize<1>(), matrix.getSize<0>(), 0, matrix.getData(), "%s",
                                   options.c_str()))) {
        assertOK();
    }

    /**
    Construct a MatrixMap from a 1-d vector of diagonal elements of a diagonal matrix

    @note The inverse transformation will always be available when constructed from a diagonal matrix.

    @param[in] diag  The diagonal elements of a diagonal matrix as a vector;
                        the number of input and output coordinates is the length of the vector.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit MatrixMap(std::vector<double> const &diag, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      // form 0 = full matrix, 1 = diagonal elements only
                      astMatrixMap(diag.size(), diag.size(), 1, diag.data(), "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~MatrixMap() {}

    /// Copy constructor: make a deep copy
    MatrixMap(MatrixMap const &) = default;
    MatrixMap(MatrixMap &&) = default;
    MatrixMap &operator=(MatrixMap const &) = delete;
    MatrixMap &operator=(MatrixMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<MatrixMap> copy() const { return std::static_pointer_cast<MatrixMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<MatrixMap, AstMatrixMap>();
    }

    /// Construct a MatrixMap from a raw AST pointer
    explicit MatrixMap(AstMatrixMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAMatrixMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a MatrixMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
