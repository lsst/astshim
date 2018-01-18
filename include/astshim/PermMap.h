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
#ifndef ASTSHIM_PERMMAP_H
#define ASTSHIM_PERMMAP_H

#include <memory>
#include <algorithm>  // for std::max
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A @ref Mapping which permutes the order of coordinates, and possibly also changes
the number of coordinates, between its input and output.

In addition to permuting the coordinate order, a PermMap may also assign constant values to coordinates.
This is useful when the number of coordinates is being increased as it allows fixed values to be assigned
to any new ones.

### Attributes

@ref PermMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class PermMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a PermMap

    Unlike AST's astPermMap, you must always provide non-empty inperm and outperm vectors.

    @param[in] inperm  A vector of `nin` elements; each element specifies the number of the
            output coordinate whose value is to be used (note that this array
            therefore defines the inverse coordinate transformation).
            Coordinates are numbered starting from 1.
    @param[in] outperm  A vector of `nout` elements; each element specifies the number of the
            input coordinate whose value is to be used (note that this array
            therefore defines the forward coordinate transformation).
            Coordinates are numbered starting from 1.
            Values may also be negative; see the `constant` parameter for details.
    @param[in] constant  An vector containing values which may be assigned to input and/or output
              coordinates instead of deriving them from other coordinate values.  If either
              of the `inperm` or `outperm` arrays contains a negative value, it is used to
              address this `constant` array (such that -1 addresses the first element, -2 addresses
              the second element, etc.)  and the value obtained is used as the corresponding
              coordinate value.
    @param[in] options  Comma-separated list of attribute assignments.

    @throws std::invalid_argument if:
    - `inperm` or `outperm` are empty
    - `inperm` or `outperm` specify a constant that is not available because `constant` has too few elements.
    */
    explicit PermMap(std::vector<int> const &inperm, std::vector<int> const &outperm,
                     std::vector<double> const &constant = {}, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(makeRawMap(inperm, outperm, constant, options))) {}

    virtual ~PermMap() {}

    /// Copy constructor: make a deep copy
    PermMap(PermMap const &) = default;
    PermMap(PermMap &&) = default;
    PermMap &operator=(PermMap const &) = delete;
    PermMap &operator=(PermMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<PermMap> copy() const { return std::static_pointer_cast<PermMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<PermMap, AstPermMap>();
    }

    /// Construct a PermMap from a raw AST pointer
    explicit PermMap(AstPermMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAPermMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a PermMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    AstPermMap *makeRawMap(std::vector<int> const &inperm, std::vector<int> const &outperm,
                           std::vector<double> const &constant = {}, std::string const &options = "");
};

}  // namespace ast

#endif
