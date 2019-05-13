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
#ifndef ASTSHIM_SKYFRAME_H
#define ASTSHIM_SKYFRAME_H

#include <memory>
#include <vector>

#include "astshim/Frame.h"

namespace ast {

/**
SkyFrame is a specialised form of Frame which describes celestial longitude/latitude coordinate systems.

The particular celestial coordinate system to be represented is specified by setting the SkyFrame's
`System` attribute (currently, the default is `ICRS`) qualified, as necessary, by a mean `Equinox` value
and/or an `Epoch`.

For each of the supported celestial coordinate systems, a SkyFrame can apply an optional
shift of origin to create a coordinate system representing offsets within the celestial
coordinate system from some specified point. This offset coordinate system can also be rotated
to define new longitude and latitude axes. See attributes `SkyRef`, `SkyRefIs` and `SkyRefP`.

All the coordinate values used by a SkyFrame are in radians. These may be formatted
in more conventional ways for display by using @ref format.

## Attributes

SkyFrame provides the following attributes, in addition to those provided by @ref Frame,
@ref Mapping and @ref Object

- @ref SkyFrame_AlignOffset "AlignOffset": align SkyFrames using the offset coordinate system?
- @ref SkyFrame_AsTime "AsTime(axis)": format celestial coordinates as times?
- @ref SkyFrame_Equinox "Equinox": epoch of the mean equinox.
- @ref SkyFrame_IsLatAxis "IsLatAxis(axis)": is the specified axis the latitude axis?
- @ref SkyFrame_IsLonAxis "IsLonAxis(axis)": is the specified axis the longitude axis?
- @ref SkyFrame_LatAxis "LatAxis": index of the latitude axis.
- @ref SkyFrame_LonAxis "LonAxis": index of the longitude axis.
- @ref SkyFrame_NegLon "NegLon": display longitude values in the range [-pi,pi]?
- @ref SkyFrame_Projection "Projection": sky projection description.
- @ref SkyFrame_SkyRef "SkyRef(axis)": position defining location of the offset coordinate system.
- @ref SkyFrame_SkyRefIs "SkyRefIs": selects the nature of the offset coordinate system.
- @ref SkyFrame_SkyRefP "SkyRefP(axis)": position defining orientation of the offset coordinate system.
- @ref SkyFrame_SkyTol "SkyTol": smallest significant shift in sky coordinates.
*/
class SkyFrame : public Frame {
    friend class Object;

public:
    /**
    Construct a SkyFrame

    @param[in] options  String containing an optional comma-separated list
        of attribute assignments to be used for initialising the new SkyFrame.
        The syntax used is identical to that for @ref set.

    ### Examples

    - `auto auto = astSkyFrame("")`

        Creates a SkyFrame to describe the default ICRS celestial coordinate system.

    - `auto auto = astSkyFrame("System = FK5, Equinox = J2005, Digits = 10")`

       Creates a SkyFrame to describe the FK5 celestial
       coordinate system, with a mean Equinox oc        Because especially accurate coordinates will be used,
       additional precision (10 digits) has been requested. This will
       be used when coordinate values are formatted for display.

    - `auto auto = astSkyFrame("System = FK4, Equinox = 1955-sep-2")`

       Creates a SkyFrame to describe the old FK4 celestial
       coordinate system.  A default Epoch value (B1950.0) is used,
       but the mean Equinox value is given explicitly as "1955-sep-2".

    - `auto auto = astSkyFrame("System = GAPPT, Epoch = J2000")`

       Creates a SkyFrame to describe the Geocentric Apparent
       celestial coordinate system.

    ### Notes

    - Currently, the default celestial coordinate system is
        `ICRS`. However, this default may change in future as new
        astrometric standards evolve. The intention is to track the most
        modern appropriate standard. For this reason, you should use the
        default only if this is what you intend (and can tolerate any
        associated slight change in behaviour with future versions of
        this function). If you intend to use the `ICRS` system indefinitely
        then you should specify it explicitly using an `options` value of `System=ICRS`.
    - Whichever celestial coordinate system is represented, it will
         have two axes.  The first of these will be the longitude axis
         and the second will be the latitude axis. This order can be
         changed using @ref permAxes if required.
    - When conversion between two SkyFrames is requested (as when
        supplying SkyFrames to @ref convert),
        account will be taken of the nature of the celestial coordinate
        systems they represent, together with any qualifying mean Equinox or
        Epoch values, etc. The @ref Frame_AlignSystem "AlignSystem" attribute will also be taken into
        account. The results will therefore fully reflect the
        relationship between positions on the sky measured in the two
        systems.
    */
    explicit SkyFrame(std::string const &options = "")
            : Frame(reinterpret_cast<AstFrame *>(astSkyFrame("%s", options.c_str()))) {
        assertOK();
    }

    virtual ~SkyFrame() {}

    /// Copy constructor: make a deep copy
    SkyFrame(SkyFrame const &) = default;
    SkyFrame(SkyFrame &&) = default;
    SkyFrame &operator=(SkyFrame const &) = delete;
    SkyFrame &operator=(SkyFrame &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SkyFrame> copy() const { return std::static_pointer_cast<SkyFrame>(copyPolymorphic()); }

    /// Get @ref SkyFrame_AlignOffset "AlignOffset": align SkyFrames using the offset coordinate system?
    bool getAlignOffset() const { return getB("AlignOffset"); }

    /// Get @ref SkyFrame_AsTime "AsTime(axis)" for one axis: format celestial coordinates as times?
    bool getAsTime(int axis) const { return getB(detail::formatAxisAttr("AsTime", axis)); }

    /// Get @ref SkyFrame_Equinox "Equinox": epoch of the mean equinox.
    double getEquinox() const { return getD("Equinox"); }

    /// Get @ref SkyFrame_IsLatAxis "IsLatAxis(axis)" for one axis: is the specified axis the latitude axis?
    bool getIsLatAxis(int axis) const { return getB(detail::formatAxisAttr("IsLatAxis", axis)); }

    /// Get @ref SkyFrame_IsLonAxis "IsLonAxis(axis)" for one axis: is the specified axis the longitude axis?
    bool getIsLonAxis(int axis) const { return getB(detail::formatAxisAttr("IsLonAxis", axis)); }

    /// Get @ref SkyFrame_LatAxis "LatAxis": index of the latitude axis.
    int getLatAxis() const { return getI("LatAxis"); }

    /// Get @ref SkyFrame_LonAxis "LonAxis": index of the longitude axis.
    int getLonAxis() const { return getI("LonAxis"); }

    /// Get @ref SkyFrame_NegLon "NegLon": display longitude values in the range [-pi,pi]?
    bool getNegLon() const { return getB("NegLon"); }

    /// Get @ref SkyFrame_Projection "Projection": sky projection description.
    std::string getProjection() const { return getC("Projection"); }

    /// Get @ref SkyFrame_SkyRef "SkyRef" for both axes:
    /// position defining location of the offset coordinate system.
    std::vector<double> getSkyRef() const {
        std::vector<double> ret;
        for (int axis = 1; axis < 3; ++axis) {
            ret.push_back(getD(detail::formatAxisAttr("SkyRef", axis)));
        }
        return ret;
    }

    /// Get @ref SkyFrame_SkyRefIs "SkyRefIs": selects the nature of the offset coordinate system.
    std::string getSkyRefIs() const { return getC("SkyRefIs"); }

    /// Get @ref SkyFrame_SkyRefP "SkyRefP" for both axes:
    /// position defining orientation of the offset coordinate system.
    std::vector<double> getSkyRefP() const {
        std::vector<double> ret;
        for (int axis = 1; axis < 3; ++axis) {
            ret.push_back(getD(detail::formatAxisAttr("SkyRefP", axis)));
        }
        return ret;
    }

    /// Get @ref SkyFrame_SkyTol "SkyTol": smallest significant shift in sky coordinates.
    double getSkyTol() const { return getD("SkyTol"); }

    /// Set @ref SkyFrame_AlignOffset "AlignOffset": align SkyFrames using the offset coordinate system?
    void setAlignOffset(bool alignOffset) { setB("AlignOffset", alignOffset); }

    /// Set @ref SkyFrame_AsTime "AsTime(axis)" for one axis: format celestial coordinates as times?
    void setAsTime(int axis, bool asTime) { setB(detail::formatAxisAttr("AsTime", axis), asTime); }

    /// Set @ref SkyFrame_Equinox "Equinox": epoch of the mean equinox.
    void setEquinox(double equinox) { setD("Equinox", equinox); }

    /// Set @ref SkyFrame_NegLon "NegLon": display longitude values in the range [-pi,pi]?
    void setNegLon(bool negLon) { setB("NegLon", negLon); }

    /// Set @ref SkyFrame_Projection "Projection": sky projection description.
    void setProjection(std::string const &projection) { setC("Projection", projection); }

    /// Set @ref SkyFrame_SkyRef "SkyRef": position defining location of the offset coordinate system.
    void setSkyRef(std::vector<double> const &skyRef) {
        detail::assertEqual(skyRef.size(), "skyRef length", 2u, "number of axes");
        for (int i = 0; i < 2; ++i) {
            setD(detail::formatAxisAttr("SkyRef", i + 1), skyRef[i]);
        }
    }

    /// Set @ref SkyFrame_SkyRefIs "SkyRefIs": selects the nature of the offset coordinate system.
    void setSkyRefIs(std::string const &skyRefIs) { setC("SkyRefIs", skyRefIs); }

    /// Set @ref SkyFrame_SkyRefP "SkyRefP": position defining orientation of the offset coordinate system.
    void setSkyRefP(std::vector<double> const &skyRefP) {
        detail::assertEqual(skyRefP.size(), "skyRefP length", 2u, "number of axes");
        for (int i = 0; i < 2; ++i) {
            setD(detail::formatAxisAttr("SkyRefP", i + 1), skyRefP[i]);
        }
    }

    /// Set @ref SkyFrame_SkyTol "SkyTol": smallest significant shift in sky coordinates.
    void setSkyTol(double skyTol) { setD("SkyTol", skyTol); }

    /**
    Get a sky offset map

    Get a @ref Mapping in which the forward transformation transforms a position in the coordinate system
    given by the System attribute of the supplied SkyFrame, into the offset coordinate system
    specified by the `SkyRef`, `SkyRefP` and `SkyRefIs` attributes of the sky frame.
    A @ref UnitMap is returned if the sky frame does not define an offset coordinate system.
    */
    std::shared_ptr<Mapping> skyOffsetMap() {
        auto *rawMap = reinterpret_cast<AstObject *>(astSkyOffsetMap(getRawPtr()));
        assertOK();
        return Object::fromAstObject<Mapping>(rawMap, false);
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SkyFrame, AstSkyFrame>();
    }

    /// Construct a SkyFrame from a raw AST pointer
    explicit SkyFrame(AstSkyFrame *rawptr) : Frame(reinterpret_cast<AstFrame *>(rawptr)) {
        if (!astIsASkyFrame(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a SkyFrame";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
