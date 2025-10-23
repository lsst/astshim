/*
 * This file is part of astshim.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <sstream>
#include <stdexcept>

#include "astshim/SplineMap.h"

namespace ast {

SplineMap::SplineMap(AstSplineMap *map) : Mapping(reinterpret_cast<AstMapping *>(map)) {
    if (!astIsASplineMap(getRawPtr())) {
        std::ostringstream os;
        os << "this is a " << getClassName() << ", which is not a SplineMap";
        throw std::invalid_argument(os.str());
    }
}

/// Make a raw AstSplineMap with a specified forward transform.
AstSplineMap *SplineMap::_makeRawSplineMap(int kx, int ky, int nx, int ny, std::vector<double> const &tx,
    std::vector<double> const &ty, std::vector<double> const &cu, std::vector<double> const &cv,
    std::string const &options) const {

    const size_t nTx = tx.size();
    const size_t nTy = ty.size();
    const size_t nCoeffsU = cu.size();
    const size_t nCoeffsV = cv.size();

    if ((kx < 0) || (ky < 0) || (nx < 0) || (ny < 0)) {
        throw std::invalid_argument("The polynomial order and number of coefficients must not be negative.");
    }
    if (((size_t)(kx + nx) != nTx) || ((size_t)(ky + ny) != nTy)) {
        throw std::invalid_argument("The length of the knot positions must equal the polynomial order plus "
            "the number of coefficients.");
    }
    if (((size_t)(nx * ny) != nCoeffsU) || ((size_t)(nx * ny) != nCoeffsV)) {
        throw std::invalid_argument("The length of the coefficients must equal product of the arguments nx "
            "and ny.");
    }

    return reinterpret_cast<AstSplineMap *>(astSplineMap(kx, ky, nx, ny, tx.data(), ty.data(), cu.data(),
        cv.data(), options.c_str()));
}

}  // namespace ast
