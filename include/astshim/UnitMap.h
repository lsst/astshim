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
#ifndef ASTSHIM_UNITMAP_H
#define ASTSHIM_UNITMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A UnitMap is a unit (null) Mapping that has no effect on the coordinates supplied to it.
They are simply copied. This can be useful if a Mapping is required (e.g. to pass to another function)
but you do not want it to have any effect.

The @ref Mapping_NIn "NIn" and @ref Mapping_NOut "NOut" attributes of a @ref UnitMap
are always equal and are specified when it is created.

### Attributes

@ref UnitMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class UnitMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref UnitMap

    @param[in] ncoord  The number of input and output coordinates (these numbers are necessarily the same).
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit UnitMap(int ncoord, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astUnitMap(ncoord, "%s", options.c_str()))) {
                assertOK();
            }

    virtual ~UnitMap() {}

    /// Copy constructor: make a deep copy
    UnitMap(UnitMap const &) = default;
    UnitMap(UnitMap &&) = default;
    UnitMap &operator=(UnitMap const &) = delete;
    UnitMap &operator=(UnitMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<UnitMap> copy() const { return std::static_pointer_cast<UnitMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<UnitMap, AstUnitMap>();
    }

    /// Construct a UnitMap from a raw AST pointer
    explicit UnitMap(AstUnitMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAUnitMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a UnitMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
