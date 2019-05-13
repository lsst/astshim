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
#ifndef ASTSHIM_SPHMAP_H
#define ASTSHIM_SPHMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A SphMap is a Mapping which transforms points from a 3-dimensional Cartesian coordinate system into
a 2-dimensional spherical coordinate system (longitude and latitude on a unit sphere centred at the origin).
It works by regarding the input coordinates as position vectors and finding their intersection with
the sphere surface. The inverse transformation always produces points which are a unit distance
from the origin (i.e. unit vectors).

### Attributes

- @ref SphMap_UnitRadius "UnitRadius": input vectors lie on a unit sphere?
- @ref SphMap_PolarLong "PolarLong": the longitude value to assign to either pole (radians).

### Notes

- The spherical coordinates are longitude (positive anti-clockwise looking from
    the positive latitude pole) and latitude. The Cartesian coordinates are right-handed,
    with the x axis (axis 1) at zero longitude and latitude, and the z axis (axis 3)
    at the positive latitude pole.
- At either pole, the longitude is set to the value of the PolarLong attribute.
- If the Cartesian coordinates are all zero, then the longitude and latitude are
  set to the value AST__BAD.
*/
class SphMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref SphMap

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit SphMap(std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astSphMap("%s", options.c_str()))) {
        assertOK();
    }

    virtual ~SphMap() {}

    /// Copy constructor: make a deep copy
    SphMap(SphMap const &) = default;
    SphMap(SphMap &&) = default;
    SphMap &operator=(SphMap const &) = delete;
    SphMap &operator=(SphMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SphMap> copy() const { return std::static_pointer_cast<SphMap>(copyPolymorphic()); }

    /// Get @ref SphMap_UnitRadius "UnitRadius": input vectors lie on a unit sphere?
    bool getUnitRadius() const { return getB("UnitRadius"); }

    /// Get @ref SphMap_PolarLong "PolarLong": the longitude value to assign to either pole (radians).
    double getPolarLong() const { return getD("PolarLong"); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SphMap, AstSphMap>();
    }

    /// Construct a SphMap from a raw AST pointer
    explicit SphMap(AstSphMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsASphMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a SphMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
