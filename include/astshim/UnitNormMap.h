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
#ifndef ASTSHIM_UNITNORMMAP_H
#define ASTSHIM_UNITNORMMAP_H

#include <memory>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
The forward transformation of a UnitNormMap subtracts the specified centre
and then transforms the resulting vector to a unit vector and the vector norm.
The output contains one more coordinate than the input: the initial NIn outputs
are in the same order as the input; the final output is the norm.

The inverse transformation of a UnitNormMap multiplies each component
of the provided vector by the provided norm and adds the specified centre.
The output contains one fewer coordinate than the input: the initial NIn inputs
are in the same order as the output; the final input is the norm.

UnitNormMap enables radially symmetric transformations, as follows:
- apply a UnitNormMap to produce a unit vector and norm (radius)
- apply a one-dimensional mapping to the norm (radius), while passing the unit vector unchanged
- apply the same UnitNormMap in the inverse direction to produce the result

### Attributes

@ref UnitNormMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class UnitNormMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref UnitNormMap

    @param[in] centre   An vector containing the values to be subtracted from the input
                        coordinates before computing unit vector and norm.
                        The length of this vector is the number of input axes of the mapping
                        (and one less than the number of outputs).
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit UnitNormMap(std::vector<double> const &centre, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      astUnitNormMap(centre.size(), centre.data(), "%s", options.c_str()))) {
                          assertOK();
                      }

    virtual ~UnitNormMap() {}

    /// Copy constructor: make a deep copy
    UnitNormMap(UnitNormMap const &) = default;
    UnitNormMap(UnitNormMap &&) = default;
    UnitNormMap &operator=(UnitNormMap const &) = delete;
    UnitNormMap &operator=(UnitNormMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<UnitNormMap> copy() const {
        return std::static_pointer_cast<UnitNormMap>(copyPolymorphic());
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<UnitNormMap, AstUnitNormMap>();
    }

    /// Construct a UnitNormMap from a raw AST pointer
    explicit UnitNormMap(AstUnitNormMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAUnitNormMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a UnitNormMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
