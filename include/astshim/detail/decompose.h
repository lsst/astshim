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
#ifndef ASTSHIM_DECOMPOSE_H
#define ASTSHIM_DECOMPOSE_H

#include <memory>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {
namespace detail {

/**
Return a deep copy of one of the two component mappings.

This utility function is intended to be exposed by classes
that need it (e.g. @ref CmpMap, @ref CmpFrame and @ref TranMap)
as `operator[]`.

@param[in] map  Mapping to decomposel
@param[in] i  Index: 0 for the first mapping, 1 for the second.

@throw std::invalid_argument if `i` is not 0 or 1.
@throw std::runtime_error if `map` does not contain the requested mapping.
*/
template<typename T, typename AstT>
std::shared_ptr<T> decompose(Mapping const & map, int i) {
    if ((i < 0) || (i > 1)) {
        std::ostringstream os;
        os << "i =" << i << "; must be 0 or 1";
        throw std::invalid_argument(os.str());
    }
    AstMapping * rawptr1;
    AstMapping * rawptr2;
    int series, invert1, invert2;
    astDecompose(map.getRawPtr(), &rawptr1, &rawptr2, &series, &invert1, &invert2);
    auto * retptr = reinterpret_cast<AstT *>(i == 0 ? rawptr1 : rawptr2);
    if (!retptr) {
        throw std::runtime_error("The requested component does not exist");
    }
    return T(retptr).copy();
}

}}  // namespace ast::detail

#endif