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
#ifndef ASTSHIM_PCDMAP_H
#define ASTSHIM_PCDMAP_H

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A PcdMap is a non-linear @ref Mapping which transforms 2-dimensional positions to correct for
the radial distortion introduced by some cameras and telescopes. This can take the form
either of pincushion or barrel distortion, and is characterized by a single distortion coefficient.
A PcdMap is specified by giving this distortion coefficient and the coordinates of the centre of
the radial distortion. The forward transformation of a PcdMap applies the distortion:

    RD = R * ( 1 + disco * R * R )

where `R` is the undistorted radial distance from `pcdcen`, the distortion centre,
`RD` is the radial distance from the same centre in the presence of distortion.

The inverse transformation of a PcdMap removes the distortion produced by the forward transformation.
The expression used to derive `R` from `RD` is an approximate inverse of the expression above,
obtained from two iterations of the Newton-Raphson method. The mismatch between the forward
and inverse expressions is negligible for astrometric applications (to reach 1 milliarcsec
at the edge of the Anglo-Australian Telescope triplet or a Schmidt field would require
field diameters of 2.4 and 42 degrees respectively).

### Attributes

In addition to those attributes provided by @ref Mapping and @ref Object,
@ref PcdMap has the following attributes:

- @anchor PcdMap_Disco `Disco`: pincushion/barrel distortion coefficient.

    The pincushion/barrel distortion coefficient.
    For pincushion distortion, the value should be positive.
    For barrel distortion, it should be negative.
    0 gives no distortion.

    Note that the forward transformation of a PcdMap applies the distortion
    specified by this attribute and the inverse transformation removes this distortion.

- @anchor PcdMap_PcdCen `PcdCen(axis)`: Centre coordinates of pincushion/barrel distortion.

*/
class PcdMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a PcdMap

    @param[in] disco   The distortion coefficient.  Negative values give barrel distortion, positive
              values give pincushion distortion, and zero gives no distortion.
    @param[in] pcdcen  A 2-element vector containing the coordinates of the centre of the distortion.
    @param[in] options  Comma-separated list of attribute assignments.

    @throws std::invalid_argument if pcdcen does not have exactly 2 elements.
    */
    PcdMap(double disco, std::vector<double> const &pcdcen, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(_makeRawPcdMap(disco, pcdcen, options))) {}

    virtual ~PcdMap() {}

    /// Copy constructor: make a deep copy
    PcdMap(PcdMap const &) = default;
    PcdMap(PcdMap &&) = default;
    PcdMap &operator=(PcdMap const &) = delete;
    PcdMap &operator=(PcdMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<PcdMap> copy() const { return std::static_pointer_cast<PcdMap>(copyPolymorphic()); }

    /// Get @ref PcdMap_Disco "Disco": pincushion/barrel distortion coefficient
    double getDisco() const { return getD("Disco"); };

    /// Get @ref PcdMap_PcdCen `PcdCen(axis)` for one axis: centre coordinates of pincushion/barrel distortion
    double getPcdCen(int axis) const { return getD(detail::formatAxisAttr("PcdCen", axis)); }

    /// Get @ref PcdMap_PcdCen `PcdCen` for both axes: centre coordinates of pincushion/barrel distortion
    std::vector<double> getPcdCen() const {
        std::vector<double> ctr;
        for (auto axis = 1; axis < 3; ++axis) {
            ctr.push_back(getPcdCen(axis));
        }
        return ctr;
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<PcdMap, AstPcdMap>();
    }

    /// Construct a PcdMap from a raw AST pointer
    explicit PcdMap(AstPcdMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAPcdMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a PcdMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    AstPcdMap *_makeRawPcdMap(double disco, std::vector<double> const &pcdcen,
                              std::string const &options = "") {
        if (pcdcen.size() != 2) {
            std::ostringstream os;
            os << "pcdcen.size() = " << pcdcen.size() << "; must be 2";
            throw std::invalid_argument(os.str());
        }
        auto result = astPcdMap(disco, pcdcen.data(), "%s", options.c_str());
        assertOK();
        return result;
    }
};

}  // namespace ast

#endif
