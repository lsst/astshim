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
#ifndef ASTSHIM_WCSMAP_H
#define ASTSHIM_WCSMAP_H

#include <memory>
#include <sstream>
#include <utility>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Mapping.h"

namespace ast {

/**
WCS types that give the projection type code (in upper case) as
used in the FITS-WCS "CTYPEi" keyword. You should consult the
FITS-WCS paper for a list of the available projections. The
additional code of WcsType::TPN can be supplied which represents a
TAN projection with polynomial correction terms as defined in an
early draft of the FITS-WCS paper.

These have the same value as the corresponding AST__ constant,
e.g. WcsType::TAN = AST__TAN.

I am not sure what AST__WCSBAD is used for, but I included it anyway.
*/
enum class WcsType {
    AZP = AST__AZP,
    SZP = AST__SZP,
    TAN = AST__TAN,
    STG = AST__STG,
    SIN = AST__SIN,
    ARC = AST__ARC,
    ZPN = AST__ZPN,
    ZEA = AST__ZEA,
    AIR = AST__AIR,
    CYP = AST__CYP,
    CEA = AST__CEA,
    CAR = AST__CAR,
    MER = AST__MER,
    SFL = AST__SFL,
    PAR = AST__PAR,
    MOL = AST__MOL,
    AIT = AST__AIT,
    COP = AST__COP,
    COE = AST__COE,
    COD = AST__COD,
    COO = AST__COO,
    BON = AST__BON,
    PCO = AST__PCO,
    TSC = AST__TSC,
    CSC = AST__CSC,
    QSC = AST__QSC,
    NCP = AST__NCP,
    GLS = AST__GLS,
    TPN = AST__TPN,
    HPX = AST__HPX,
    XPH = AST__XPH,
    WCSBAD = AST__WCSBAD,
};

/**
Map from a spherical system to a cartesian system using standard FITS sky coordinate projections.

@ref WcsMap is used to represent sky coordinate projections as
described in the FITS world coordinate system (FITS-WCS) paper II
"Representations of Celestial Coordinates in FITS" by M. Calabretta
and E.W. Griesen. This paper defines a set of functions, or sky
projections, which transform longitude-latitude pairs representing
spherical celestial coordinates into corresponding pairs of Cartesian
coordinates (and vice versa).

A @ref WcsMap is a specialised form of Mapping which implements these
sky projections and applies them to a specified pair of coordinates.
All the projections in the FITS-WCS paper are supported, plus the now
deprecated "TAN with polynomial correction terms" projection which
is refered to here by the code "TPN". Using the FITS-WCS terminology,
the transformation is between "native spherical" and "projection
plane" coordinates (also called "intermediate world coordinates".
These coordinates may, optionally, be embedded in a space with more
than two dimensions, the remaining coordinates being copied unchanged.
Note, however, that for consistency with other AST facilities, a
@ref WcsMap handles coordinates that represent angles in radians (rather
than the degrees used by FITS-WCS).

The type of FITS-WCS projection to be used and the coordinates
(axes) to which it applies are specified when a @ref WcsMap is first
created. The projection type may subsequently be determined
using the @ref WcsMap_WcsType "WcsType" attribute and the coordinates on which it acts
may be determined using the @ref WcsMap_WcsType "WcsAxis(lonlat)" attribute.

Each @ref WcsMap also allows up to 100 "projection parameters" to be
associated with each axis. These specify the precise form of the
projection, and are accessed using @ref WcsMap_PVi_m "PVi_m" attribute,
where "i" is the integer axis index (starting at 1), and "m" is an integer
"parameter index" in the range 0 to 99. The number of projection
parameters required by each projection, and their meanings, are
dependent upon the projection type (most projections either do not
use any projection parameters, or use parameters 1 and 2 associated
with the latitude axis). Before creating a @ref WcsMap you should consult
the FITS-WCS paper for details of which projection parameters are
required, and which have defaults. When creating the @ref WcsMap, you must
explicitly set values for all those required projection parameters
which do not have defaults defined in this paper.

### Attributes

- @ref WcsMap_NatLat "NatLat": native latitude of the reference point of a FITS-WCS projection.
- @ref WcsMap_NatLon "NatLon": native longitude of the reference point of a FITS-WCS projection.
- @ref WcsMap_PVi_m "PVi_m": FITS-WCS projection parameters.
- @ref WcsMap_PVMax "PVMax": maximum number of FITS-WCS projection parameters.
- @ref WcsMap_WcsAxis "WcsAxis(lonlat)": FITS-WCS projection axes.
- @ref WcsMap_WcsType "WcsType": FITS-WCS projection type.

### Notes

- The forward transformation of a WcsMap converts between
FITS-WCS "native spherical" and "relative physical" coordinates,
while the inverse transformation converts in the opposite
direction.
- If any set of coordinates cannot be transformed (for example,
many projections do not cover the entire celestial sphere), then
a WcsMap will yield coordinate values of `nan`.
*/
class WcsMap : public Mapping {
    friend class Object;

public:
    /**
    Create a WcsMap

    @param[in] ncoord  The number of coordinate values for each point to be
                        transformed (i.e. the number of dimensions of the space in
                        which the points will reside). This must be at least 2. The
                        same number is applicable to both input and output points.
    @param[in] type  The type of FITS-WCS projection to apply, as a WcsType enum
                        such as WcsType::TAN (for a tangent
                        plane projection). The enum constant name
                        give the projection type code (in upper case) as
                        used in the FITS-WCS "CTYPEi" keyword. You should consult the
                        FITS-WCS paper for a list of the available projections. The
                        additional code of WcsType::TPN can be supplied which represents a
                        TAN projection with polynomial correction terms as defined in an
                        early draft of the FITS-WCS paper.
    @param[in] lonax  Index of the longitude axis. This should lie in the range 1 to `ncoord`.
    @param[in] latax  Index of the latitude axis. This should lie in the range 1 to `ncoord`.
    @param[in] options  Pointer to a null-terminated string containing an optional
                        comma-separated list of attribute assignments to be used for
                        initialising the new WcsMap. The syntax used is identical to
                        that for the astSet function and may include "printf" format
                        specifiers identified by "%" symbols in the normal way.
                        If the sky projection to be implemented requires projection
                        parameter values to be set, then this should normally be done
                        here via the PVi_m attribute (see the "Examples"
                        section). Setting values for these parameters is mandatory if
                        they do not have default values (as defined in the FITS-WCS
                        paper).

    @warning The validity of any projection parameters given via the @ref WcsMap_PVi_m PVi_m
    parameter in the "options" string is not checked at construction.
    However, their validity is checked when the resulting
    WcsMap is used to transform coordinates, and an error will
    result if the projection parameters do not satisfy all the
    required constraints (as defined in the FITS-WCS paper).

    ### Examples

    - `auto wcsmap = ast::WcsMap(2, WcsType::MER, 1, 2, "")`

        Create a WcsMap that implements a FITS-WCS Mercator
        projection on pairs of coordinates, with coordinates 1 and 2
        representing the longitude and latitude respectively. Note
        that the FITS-WCS Mercator projection does not require any
        projection parameters.

    - `auto wcsmap = ast::WcsMap(3, WcsType::COE, 2, 3, "PV3_1=40.0")`

        Create a WcsMap that implements a FITS-WCS conical equal
        area projection. The WcsMap acts on points in a 3-dimensional
        space; coordinates 2 and 3 represent longitude and latitude
        respectively, while the values of coordinate 1 are copied
        unchanged.  Projection parameter 1 associatyed with the latitude
        axis (corresponding to FITS keyword "PV3_1") is required and has
        no default, so is set explicitly to 40.0 degrees. Projection
        parameter 2 (corresponding to FITS keyword "PV3_2") is required
        but has a default of zero, so need not be specified.
    */
    explicit WcsMap(int ncoord, WcsType type, int lonax, int latax, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      astWcsMap(ncoord, static_cast<int>(type), lonax, latax, "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~WcsMap() {}

    /// Copy constructor: make a deep copy
    WcsMap(WcsMap const &) = default;
    WcsMap(WcsMap &&) = default;
    WcsMap &operator=(WcsMap const &) = delete;
    WcsMap &operator=(WcsMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<WcsMap> copy() const { return std::static_pointer_cast<WcsMap>(copyPolymorphic()); }

    /// get @ref WcsMap_NatLat "NatLat": native latitude of the reference point of a FITS-WCS projection.
    double getNatLat() const { return getD("NatLat"); }

    /// get @ref WcsMap_NatLon "NatLon": native longitude of the reference point of a FITS-WCS projection.
    double getNatLon() const { return getD("NatLon"); }

    /**
    Get @ref WcsMap_PVi_m "PVi_m" for one value of i and m: a FITS-WCS projection parameter.

    @param[in] i  Axis index, starting at 1
    @param[in] m  Parameter number, in range 1 to @ref WcsMap_PVMax "PVMax"
    */
    double getPVi_m(int i, int m) const {
        std::stringstream os;
        os << "PV" << i << "_" << m;
        return getD(os.str());
    }

    /// Get @ref WcsMap_PVMax "PVMax(axis)" for one axis: maximum number of FITS-WCS projection parameters.
    int getPVMax(int axis) const { return getI(detail::formatAxisAttr("PVMax", axis)); }

    /**
    Get @ref WcsMap_WcsAxis FITS-WCS projection axis for longitude, latitude.
    */
    std::pair<int, int> getWcsAxis() const {
        return std::make_pair(getI(detail::formatAxisAttr("WcsAxis", 1)),
                              getI(detail::formatAxisAttr("WcsAxis", 2)));
    }

    /// Get @ref WcsMap_WcsType "WcsType": FITS-WCS projection type.
    WcsType getWcsType() const { return static_cast<WcsType>(getI("WcsType")); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<WcsMap, AstWcsMap>();
    }

    /// Construct a WcsMap from a raw AST pointer
    explicit WcsMap(AstWcsMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAWcsMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a WcsMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
