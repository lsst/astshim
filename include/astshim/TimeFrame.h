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
#ifndef ASTSHIM_TIMEFRAME_H
#define ASTSHIM_TIMEFRAME_H

#include <memory>
#include <vector>

#include "astshim/detail/utils.h"
#include "astshim/Frame.h"

namespace ast {

/**
A TimeFrame is a specialised form of one-dimensional Frame which
represents various coordinate systems used to describe positions in
time.

A TimeFrame represents a moment in time as either an Modified Julian
Date (MJD), a Julian Date (JD), a Besselian epoch or a Julian epoch,
as determined by the System attribute. Optionally, a zero point can be
specified (using attribute TimeOrigin) which results in the TimeFrame
representing time offsets from the specified zero point.

Even though JD and MJD are defined as being in units of days, the
TimeFrame class allows other units to be used (via the Unit attribute)
on the basis of simple scalings (60 seconds = 1 minute, 60 minutes = 1
hour, 24 hours = 1 day, 365.25 days = 1 year). Likewise, Julian epochs
can be described in units other than the usual years. Besselian epoch
are always represented in units of (tropical) years.

The TimeScale attribute allows the time scale to be specified (that
is, the physical process used to define the rate of flow of time).
MJD, JD and Julian epoch can be used to represent a time in any
supported time scale. However, Besselian epoch may only be used with the
"TT" (Terrestrial Time) time scale. The list of supported time scales
includes universal time and siderial time. Strictly, these represent
angles rather than time scales, but are included in the list since
they are in common use and are often thought of as time scales.
When a time value is formatted it can be formated either as a simple
floating point value, or as a Gregorian date (see the Format
attribute).

### Attributes

TimeFrame has the following attributes in addition
to those provided by @ref Frame, @ref Mapping and @ref Object

- @ref TimeFrame_AlignTimeScale "AlignTimeScale": time scale in which to align TimeFrames.
- @ref TimeFrame_LTOffset "LTOffset": the offset of Local Time from UTC, in hours.
- @ref TimeFrame_TimeOrigin "TimeOrigin": the zero point for TimeFrame axis values.
- @ref TimeFrame_TimeScale "TimeScale": the timescale used by the TimeFrame.

Several of the Frame attributes inherited by the TimeFrame class
refer to a specific axis of the Frame (for instance Unit(axis),
Label(axis), etc). Since a TimeFrame is strictly one-dimensional,
it allows these attributes to be specified without an axis index.
So for instance, "Unit" is allowed in place of "Unit(1)".

*/
class TimeFrame : public Frame {
    friend class Object;

public:
    /**
    Construct a TimeFrame

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit TimeFrame(std::string const &options = "")
            : Frame(reinterpret_cast<AstFrame *>(astTimeFrame("%s", options.c_str()))) {
                assertOK();
            }

    virtual ~TimeFrame() {}

    /// Copy constructor: make a deep copy
    TimeFrame(TimeFrame const &) = default;
    TimeFrame(TimeFrame &&) = default;
    TimeFrame &operator=(TimeFrame const &) = delete;
    TimeFrame &operator=(TimeFrame &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<TimeFrame> copy() const {
        return std::static_pointer_cast<TimeFrame>(copyPolymorphic());
    }

    /**
    Get the current system time

    @return the current system time, as specified by the frame's attributes
        `System`, `TimeOrigin`, `LTOffset`, `TimeScale`, and `Unit`.

    @throws std::runtime_error if the frame has a `TimeScale` value which cannot be converted to TAI
        (e.g. " angular" systems such as UT1, GMST, LMST and LAST).

    ### Notes:
    - Resolution is one second.
    - This method assumes that the system time (returned by the C `time()` function)
        follows the POSIX standard, representing a continuous monotonic increasing count of SI seconds
        since the epoch `00:00:00 UTC 1 January 1970 AD` (equivalent to TAI with a constant offset).
    - Any inaccuracy in the system clock will be reflected in the value returned by this function.
    */
    double currentTime() const {
        auto result = detail::safeDouble(astCurrentTime(getRawPtr()));
        assertOK();
        return result;
    }

    /// Get @ref TimeFrame_AlignTimeScale "AlignTimeScale": time scale in which to align TimeFrames.
    std::string getAlignTimeScale() const { return getC("AlignTimeScale"); }

    /// Get @ref TimeFrame_LTOffset "LTOffset": the offset of Local Time from UTC, in hours.
    double getLTOffset() const { return getD("LTOffset"); }

    /// Get @ref TimeFrame_TimeOrigin "TimeOrigin": the zero point for TimeFrame axis values.
    double getTimeOrigin() const { return getD("TimeOrigin"); }

    /// Get @ref TimeFrame_TimeScale "TimeScale": the timescale used by the TimeFrame.
    std::string getTimeScale() const { return getC("TimeScale"); }

    /// Set @ref TimeFrame_AlignTimeScale "AlignTimeScale": time scale in which to align TimeFrames.
    void setAlignTimeScale(std::string const &scale) { return setC("AlignTimeScale", scale); }

    /// Set @ref TimeFrame_LTOffset "LTOffset": the offset of Local Time from UTC, in hours.
    void setLTOffset(double offset) { return setD("LTOffset", offset); }

    /// Set @ref TimeFrame_TimeOrigin "TimeOrigin": the zero point for TimeFrame axis values.
    void setTimeOrigin(double origin) { return setD("TimeOrigin", origin); }

    /// Set @ref TimeFrame_TimeScale "TimeScale": the timescale used by the TimeFrame.
    void setTimeScale(std::string const &scale) { return setC("TimeScale", scale); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<TimeFrame, AstTimeFrame>();
    }

    /// Construct a TimeFrame from a raw AST pointer
    explicit TimeFrame(AstTimeFrame *rawptr) : Frame(reinterpret_cast<AstFrame *>(rawptr)) {
        if (!astIsATimeFrame(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a TimeFrame";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
