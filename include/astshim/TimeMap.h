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
#ifndef ASTSHIM_TIMEMAP_H
#define ASTSHIM_TIMEMAP_H

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A TimeMap is a specialised form of 1-dimensional Mapping which can be
used to represent a sequence of conversions between standard time
coordinate systems.
*/
class TimeMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a TimeMap

    When a TimeMap is first created, it simply performs a unit (null) Mapping.
    Using the @ref add method, a series of coordinate conversion steps may then be
    added. This allows multi-step conversions between a variety of
    time coordinate systems to be assembled out of a set of building blocks.

    For details of the individual coordinate conversions available, see @ref add.

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit TimeMap(std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astTimeMap(0, "%s", options.c_str()))) {
                assertOK();
            }

    virtual ~TimeMap() {}

    /// Copy constructor: make a deep copy
    TimeMap(TimeMap const &) = default;
    TimeMap(TimeMap &&) = default;
    TimeMap &operator=(TimeMap const &) = delete;
    TimeMap &operator=(TimeMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<TimeMap> copy() const { return std::static_pointer_cast<TimeMap>(copyPolymorphic()); }

    /**
    Add one of the standard time coordinate system conversions listed below.

    When a @ref TimeMap is first created, it simply performs a unit (null) Mapping.
    By using `add` (repeatedly if necessary), one or more coordinate conversion steps may then
    be added, which the @ref TimeMap will perform in sequence. This allows
    multi-step conversions between a variety of time coordinate
    systems to be assembled out of the building blocks provided by
    this class.

    Normally, if a @ref TimeMap's Invert attribute is zero (the default),
    then its forward transformation is performed by carrying out
    each of the individual coordinate conversions specified by
    `add` in the order given (i.e. with the most recently added
    conversion applied last).

    This order is reversed if the @ref TimeMap's Invert attribute is
    non-zero (or if the inverse transformation is requested by any
    other means) and each individual coordinate conversion is also
    replaced by its own inverse. This process inverts the overall
    effect of the @ref TimeMap. In this case, the first conversion to be
    applied would be the inverse of the one most recently added.

    @param[in] cvt  String which identifies the time coordinate conversion to be added to the
       @ref TimeMap. See "Available Conversions" for details of those available.
    @param[in] args  An array containing argument values for the time
       coordinate conversion. The number of arguments required, and
       hence the number of array elements used, depends on the
       conversion specified (see the "Available Conversions"
       section). This array is ignored and an empty vector may be provided
       if no arguments are needed.

    ### Attributes

    @ref TimeMap has no attributes beyond those provided by @ref Mapping and @ref Object.

    ### Notes

    - When assembling a multi-stage conversion, it can sometimes be
    difficult to determine the most economical conversion path. A solution
    to this is to include all the steps which are (logically) necessary,
    but then to use
    astSimplify to simplify the resulting
    @ref TimeMap. The simplification process will eliminate any steps
    which turn out not to be needed.
    - This function does not check to ensure that the sequence of
    coordinate conversions added to a @ref TimeMap is physically
    meaningful.

    ### Available Conversions

    The following strings (which are case-insensitive) may be supplied
    via the "cvt" parameter to indicate which time coordinate
    conversion is to be added to the @ref TimeMap.

    The `cvt` string is followed by the number of arguments required
    in the `args` array and the description of those arguments in parenthesis.
    Units and argument names are described at the end of
    the list of conversions, and "MJD" means Modified Julian Date.

    - "MJDTOMJD"  (MJDOFF1,MJDOFF2): Convert MJD from one offset to another.
    - "MJDTOJD"  (MJDOFF,JDOFF): Convert MJD to Julian Date.
    - "JDTOMJD"  (JDOFF,MJDOFF): Convert Julian Date to MJD.
    - "MJDTOBEP" (MJDOFF,BEPOFF): Convert MJD to Besselian epoch.
    - "BEPTOMJD" (BEPOFF,MJDOFF): Convert Besselian epoch to MJD.
    - "MJDTOJEP" (MJDOFF,JEPOFF): Convert MJD to Julian epoch.
    - "JEPTOMJD" (JEPOFF,MJDOFF): Convert Julian epoch to MJD.
    - "TAITOUTC" (MJDOFF,DTAI): Convert a TAI MJD to a UTC MJD.
    - "UTCTOTAI" (MJDOFF,DTAI): Convert a UTC MJD to a TAI MJD.
    - "TAITOTT"  (MJDOFF): Convert a TAI MJD to a TT MJD.
    - "TTTOTAI"  (MJDOFF): Convert a TT MJD to a TAI MJD.
    - "TTTOTDB"  (MJDOFF,OBSLON,OBSLAT,OBSALT,DTAI): Convert a TT MJD to a TDB MJD.
    - "TDBTOTT"  (MJDOFF,OBSLON,OBSLAT,OBSALT,DTAI): Convert a TDB MJD to a TT MJD.
    - "TTTOTCG"  (MJDOFF): Convert a TT MJD to a TCG MJD.
    - "TCGTOTT"  (MJDOFF): Convert a TCG MJD to a TT MJD.
    - "TDBTOTCB" (MJDOFF): Convert a TDB MJD to a TCB MJD.
    - "TCBTOTDB" (MJDOFF): Convert a TCB MJD to a TDB MJD.
    - "UTTOGMST" (MJDOFF): Convert a UT MJD to a GMST MJD.
    - "GMSTTOUT" (MJDOFF): Convert a GMST MJD to a UT MJD.
    - "GMSTTOLMST" (MJDOFF,OBSLON,OBSLAT): Convert a GMST MJD to a LMST MJD.
    - "LMSTTOGMST" (MJDOFF,OBSLON,OBSLAT): Convert a LMST MJD to a GMST MJD.
    - "LASTTOLMST" (MJDOFF,OBSLON,OBSLAT): Convert a GMST MJD to a LMST MJD.
    - "LMSTTOLAST" (MJDOFF,OBSLON,OBSLAT): Convert a LMST MJD to a GMST MJD.
    - "UTTOUTC" (DUT1): Convert a UT1 MJD to a UTC MJD.
    - "UTCTOUT" (DUT1): Convert a UTC MJD to a UT1 MJD.
    - "LTTOUTC" (LTOFF): Convert a Local Time MJD to a UTC MJD.
    - "UTCTOLT" (LTOFF): Convert a UTC MJD to a Local Time MJD.

    The units for the values processed by the above conversions are as follows:

    - Julian epochs and offsets: Julian years
    - Besselian epochs and offsets: Tropical years
    - Modified Julian Dates and offsets: days
    - Julian Dates and offsets: days

    The arguments used in the above conversions are the zero-points
    used by the @ref Mapping.applyForward function.
    The axis values supplied and returned by
    @ref Mapping.applyForward are offsets away from these zero-points:

    - `MJDOFF`: The zero-point being used with MJD values.
    - `JDOFF`: The zero-point being used with Julian Date values.
    - `BEPOFF`: The zero-point being used with Besselian epoch values.
    - `JEPOFF`: The zero-point being used with Julian epoch values.
    - `OBSLON`: Observer longitude in radians (+ve westwards).
    - `OBSLAT`: Observer geodetic latitude (IAU 1975) in radians (+ve northwards).
    - `OBSALT`: Observer geodetic altitude (IAU 1975) in metres.
    - `DUT1`: The UT1-UTC value to use.
    - `LTOFF`: The offset between Local Time and UTC (in hours, positive
    for time zones east of Greenwich).
    */
    void add(std::string const &cvt, std::vector<double> const &args) {
        astTimeAdd(getRawPtr(), cvt.c_str(), args.size(), args.data());
        assertOK();
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<TimeMap, AstTimeMap>();
    }

    /// Construct a TimeMap from a raw AST pointer
    explicit TimeMap(AstTimeMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsATimeMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a TimeMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
