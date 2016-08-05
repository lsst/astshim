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
#ifndef ASTSHIM_SHIFTMAP_H
#define ASTSHIM_SHIFTMAP_H

#include <algorithm>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
ShiftMap is a linear @ref Mapping which shifts each axis by a specified constant value.

### Attributes

@ref ShiftMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class ShiftMap: public Mapping {
friend class Object;
public:
    /** Construct a @ref ShiftMap

    @param[in] shift  Offset to be added to the input coordinates to obtain the output coordinates.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit ShiftMap(std::vector<double> const & shift, std::string const & options="") :
        Mapping(reinterpret_cast<AstMapping *>(astShiftMap(shift.size(), shift.data(), options.c_str())))
    {}

    /// Cast an object to a ShiftMap if possible, else throw std::runtime_error
    explicit ShiftMap(Object & obj) : ShiftMap(detail::shallowCopy<AstShiftMap>(obj.getRawPtr())) {}

    virtual ~ShiftMap() {}

    ShiftMap(ShiftMap const &) = default;
    ShiftMap(ShiftMap &&) = default;
    ShiftMap & operator=(ShiftMap const &) = default;
    ShiftMap & operator=(ShiftMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<ShiftMap> copy() const { return _copy<ShiftMap, AstShiftMap>(); }

private:
    /// Construct a ShiftMap from a raw AST pointer
    explicit ShiftMap(AstShiftMap * rawptr) :
        Mapping(reinterpret_cast<AstMapping *>(rawptr))
    {
        if (!astIsAShiftMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClass() << ", which is not a ShiftMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
