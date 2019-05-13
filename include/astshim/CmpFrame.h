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
#ifndef ASTSHIM_CMPFRAME_H
#define ASTSHIM_CMPFRAME_H

#include <memory>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Frame.h"

namespace ast {

/**
A CmpFrame is a compound Frame which allows two component Frames (of any class)
to be merged together to form a more complex Frame.

The axes of the two component Frames then appear together
in the resulting CmpFrame (those of the first Frame, followed by
those of the second Frame).

Since a CmpFrame is itself a Frame, it can be used as a
component in forming further CmpFrames. Frames of arbitrary
complexity may be built from simple individual Frames in this
way.

### Attributes

@ref CmpFrame has no attributes beyond those provided by @ref Frame and @ref Object.
However, the attributes of the component Frames can be accessed as if they were attributes
of the CmpFrame. For instance, if a CmpFrame contains a SpecFrame
and a SkyFrame, then the CmpFrame will recognise the "Equinox"
attribute and forward access requests to the component SkyFrame.
Likewise, it will recognise the "RestFreq" attribute and forward
access requests to the component SpecFrame. An axis index can
optionally be appended to the end of any attribute name, in which
case the request to access the attribute will be forwarded to the
primary Frame defining the specified axis.
*/
class CmpFrame : public Frame {
    friend class Object;

public:
    /**
    Construct a CmpFrame

    @param[in] frame1  First frame, describing the lower numbered coordinates.
    @param[in] frame2  Second frame.
    @param[in] options  Comma-separated list of attribute assignments.

    @warning @ref CmpFrame contains shallow copies of the provided frames (just like AST).
    If you deep copies then provide deep copies to this constructor.
    */
    explicit CmpFrame(Frame const &frame1, Frame const &frame2, std::string const &options = "")
            : Frame(reinterpret_cast<AstFrame *>(astCmpFrame(const_cast<AstObject *>(frame1.getRawPtr()),
                                                             const_cast<AstObject *>(frame2.getRawPtr()),
                                                             "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~CmpFrame() {}

    /// Copy constructor: make a deep copy
    CmpFrame(CmpFrame const &) = default;
    CmpFrame(CmpFrame &&) = default;
    CmpFrame &operator=(CmpFrame const &) = delete;
    CmpFrame &operator=(CmpFrame &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<CmpFrame> copy() const { return std::static_pointer_cast<CmpFrame>(copyPolymorphic()); }

    /**
    Return a shallow copy of one of the two component frames.

    @param[in] i  Index: 0 for the first frame, 1 for the second.
    @throws std::invalid_argument if `i` is not 0 or 1.
    */
    std::shared_ptr<Frame> operator[](int i) const { return decompose<Frame>(i, false); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<CmpFrame, AstCmpFrame>();
    }

    /// Construct a CmpFrame from a raw AST pointer
    /// (protected instead of private so that SeriesMap and ParallelMap can call it)
    explicit CmpFrame(AstCmpFrame *rawptr) : Frame(reinterpret_cast<AstFrame *>(rawptr)) {
        if (!astIsACmpFrame(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a CmpFrame";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
