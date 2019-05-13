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
#ifndef ASTSHIM_ZOOMMAP_H
#define ASTSHIM_ZOOMMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A Mapping which "zooms" a set of points about the origin by multiplying all coordinate values
by the same scale factor.

The inverse transformation is performed by dividing by this scale factor.

### Attributes

In addition to those attributes provided by @ref Mapping and @ref Object,
@ref ZoomMap has the following attributes:

- @anchor ZoomMap_Zoom `Zoom`: scale factor.
*/
class ZoomMap : public Mapping {
    friend class Object;

public:
    /**
    Create a ZoomMap

    @param[in] ncoord  The number of coordinate values for each point to be transformed (i.e.  the number
              of dimensions of the space in which the points will reside).  The same number
              is applicable to both input and output points.
    @param[in] zoom  Initial scale factor by which coordinate values should be multiplied (by the forward
              transformation) or divided (by the inverse transformation).  This factor may subsequently
              be changed via the ZoomMap's `Zoom` attribute. It may be positive or negative,
              but should not be zero.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit ZoomMap(int ncoord, double zoom, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astZoomMap(ncoord, zoom, "%s", options.c_str()))) {
                assertOK();
            }

    virtual ~ZoomMap() {}

    /// Copy constructor: make a deep copy
    ZoomMap(ZoomMap const &) = default;
    ZoomMap(ZoomMap &&) = default;
    ZoomMap &operator=(ZoomMap const &) = delete;
    ZoomMap &operator=(ZoomMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<ZoomMap> copy() const { return std::static_pointer_cast<ZoomMap>(copyPolymorphic()); }

    /// Get @ref ZoomMap_Zoom "Zoom": scale factor
    double getZoom() const { return getF("Zoom"); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<ZoomMap, AstZoomMap>();
    }

    /// Construct a ZoomMap from a raw AST pointer
    explicit ZoomMap(AstZoomMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAZoomMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a ZoomMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
