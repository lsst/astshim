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
#ifndef ASTSHIM_SLAMAP_H
#define ASTSHIM_SLAMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
SlaMap is a specialised form of @ref Mapping which can be used to represent a sequence of conversions
between standard celestial (longitude, latitude) coordinate systems.

When an SlaMap is first created, it simply performs a unit (null) @ref Mapping on a pair of coordinates.
By calling @ref add, a series of coordinate conversion steps may then be added,
selected from those provided by the SLALIB Positional Astronomy Library (Starlink User Note SUN/67).
This allows multi-step conversions between a variety of celestial coordinate systems to be assembled
out of the building blocks provided by SLALIB.

For details of the individual coordinate conversions available, see the description of the
@ref add method.

### Notes

- All coordinate values processed by an SlaMap are in radians. The first coordinate is the
    celestial longitude and the second coordinate is the celestial latitude.
- When assembling a multi-stage conversion, it can sometimes be difficult to determine
    the most economical conversion path. For example, converting to the standard
    FK5 coordinate system as an intermediate stage is often sensible in formulating
    the problem, but may introduce unnecessary extra conversion steps. A solution
    to this is to include all the steps which are (logically) necessary, but then
    to use astSimplify to simplify the resulting SlaMap.  The simplification process
    will eliminate any steps which turn out not to be needed.
*/
class SlaMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a SlaMap

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit SlaMap(std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astSlaMap(0, "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~SlaMap() {}

    /// Copy constructor: make a deep copy
    SlaMap(SlaMap const &) = default;
    SlaMap(SlaMap &&) = default;
    SlaMap &operator=(SlaMap const &) = delete;
    SlaMap &operator=(SlaMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SlaMap> copy() const { return std::static_pointer_cast<SlaMap>(copyPolymorphic()); }

    /**
    Add one of the standard celestial coordinate system conversions provided by the SLALIB
    Positional Astronomy Library (Starlink User Note SUN/67) to an existing SlaMap.

    @param[in] cvt  String which identifies the celestial coordinate conversion to be added
        to the `SlaMap`. See the "SLALIB Conversions" section for details of those available.
    @param[in] args  A vector containing argument values for the celestial coordinate conversion.
        The number of arguments required depends on the conversion specified
        (see the " SLALIB Conversions" section).  This vector is ignored if no arguments are needed.

    When an `SlaMap` is first created, it simply performs a unit (null) @ref Mapping. By calling
    @ref add (repeatedly if necessary), one or more coordinate conversion steps may then be added,
    which the SlaMap will perform in sequence. This allows multi-step conversions
    between a variety of celestial coordinate systems to be assembled out of the building blocks
    provided by SLALIB.

    The forward transformation is performed by carrying out each of the individual coordinate conversions
    specified by astSlaAdd in the order given (i.e. with the most recently added conversion applied last).
    The order is reversed in the inverse direction and each individual coordinate conversion is also
    replaced by its own inverse. This process inverts the overall effect of the `SlaMap`.
    Thus in the reverse direction the first conversion to be applied would be the inverse of the
    conversion most recently added.

    ### Notes

    - All coordinate values processed by an SlaMap are in radians. The first coordinate is the
        celestial longitude and the second coordinate is the celestial latitude.
    - When assembling a multi-stage conversion, it can sometimes be difficult to determine
        the most economical conversion path. For example, converting to the standard
        FK5 coordinate system as an intermediate stage is often sensible in formulating
        the problem, but may introduce unnecessary extra conversion steps. A solution
        to this is to include all the steps which are (logically) necessary, but then
        to use astSimplify to simplify the resulting SlaMap.  The simplification process
        will eliminate any steps which turn out not to be needed.
    - This function does not check to ensure that the sequence of coordinate conversions
        added to an @ref SlaMap is physically meaningful.

     ## SLALIB Conversions

    The following strings (which are case-insensitive) may be supplied via the `cvt` parameter
    to indicate which celestial coordinate conversion is to be added to the SlaMap.  Each
    string is derived from the name of the SLALIB routine that performs the conversion
    and the relevant documentation (SUN/67) should be consulted for details.  Where arguments
    are needed by the conversion, they are listed in parentheses.  Values for these arguments
    should be given, via the `args` vector, in the order indicated.  The argument names
    match the corresponding SLALIB routine arguments and their values should be given using
    exactly the same units, time scale, calendar, etc.  as described in SUN/67:
    - "ADDET" (EQ): Add E-terms of aberration.
    - "SUBET" (EQ): Subtract E-terms of aberration.
    - "PREBN" (BEP0,BEP1): Apply Bessel-Newcomb pre-IAU 1976 (FK4) precession model.
    - "PREC" (EP0,EP1): Apply IAU 1975 (FK5) precession model.
    - "FK45Z" (BEPOCH): Convert FK4 to FK5 (no proper motion or parallax).
    - "FK54Z" (BEPOCH): Convert FK5 to FK4 (no proper motion or parallax).
    - "AMP" (DATE,EQ): Convert geocentric apparent to mean place.
    - "MAP" (EQ,DATE): Convert mean place to geocentric apparent.
    - "ECLEQ" (DATE): Convert ecliptic coordinates to FK5 J2000.0 equatorial.
    - "EQECL" (DATE): Convert equatorial FK5 J2000.0 to ecliptic coordinates.
    - "GALEQ" : Convert galactic coordinates to FK5 J2000.0 equatorial.
    - "EQGAL" : Convert FK5 J2000.0 equatorial to galactic coordinates.
    - "HFK5Z" (JEPOCH): Convert ICRS coordinates to FK5 J2000.0 equatorial.
    - "FK5HZ" (JEPOCH): Convert FK5 J2000.0 equatorial coordinates to ICRS.
    - "GALSUP" : Convert galactic to supergalactic coordinates.
    - "SUPGAL" : Convert supergalactic coordinates to galactic.
    - "J2000H" : Convert dynamical J2000.0 to ICRS.
    - "HJ2000" : Convert ICRS to dynamical J2000.0.
    - "R2H" (LAST): Convert RA to Hour Angle.
    - "H2R" (LAST): Convert Hour Angle to RA.

    For example, to use the "ADDET" conversion, which takes a single argument EQ, you
    should consult the documentation for the SLALIB routine SLA_ADDET. This describes the
    conversion in detail and shows that EQ is the Besselian epoch of the mean equator and
    equinox.  This value should then be supplied to @ref add in args[0].

    In addition the following strings may be supplied for more complex conversions which
    do not correspond to any one single SLALIB routine (DIURAB is the magnitude of the
    diurnal aberration vector in units of "day/(2.PI)" , DATE is the Modified Julian Date
    of the observation, and (OBSX,OBSY,OBZ) are the Heliocentric-Aries-Ecliptic cartesian
    coordinates, in metres, of the observer):

    - "HPCEQ" (DATE,OBSX,OBSY,OBSZ): Convert Helioprojective-Cartesian coordinates to J2000.0 equatorial.
    - "EQHPC" (DATE,OBSX,OBSY,OBSZ): Convert J2000.0 equatorial coordinates to Helioprojective-Ca
    - "HPREQ" (DATE,OBSX,OBSY,OBSZ): Convert Helioprojective-Radial coordinates to J2000.0 equatorial.
    - "EQHPR" (DATE,OBSX,OBSY,OBSZ): Convert J2000.0 equatorial coordinates to Helioprojective-Ra
    - "HEEQ" (DATE): Convert helio-ecliptic coordinates to J2000.0 equatorial.
    - "EQHE" (DATE): Convert J2000.0 equatorial coordinates to helio-ecliptic.
    - "H2E" (LAT,DIRUAB): Convert horizon coordinates to equatorial.
    - "E2H" (LAT,DIURAB): Convert equatorial coordinates to horizon.

    Note, the "H2E" and "E2H" conversions convert between topocentric horizon coordinates
    (azimuth,elevation), and apparent local equatorial coordinates (hour angle,declination).
    Thus, the effects of diurnal aberration are taken into account in the conversions but
    the effects of atmospheric refraction are not.
    */
    void add(std::string const &cvt, std::vector<double> const &args = std::vector<double>()) {
        astSlaAdd(getRawPtr(), cvt.c_str(), args.size(), args.data());
        assertOK();
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SlaMap, AstSlaMap>();
    }

    /// Construct a SlaMap from a raw AST pointer
    explicit SlaMap(AstSlaMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsASlaMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a SlaMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
