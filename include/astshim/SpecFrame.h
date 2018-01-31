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
#ifndef ASTSHIM_SPECFRAME_H
#define ASTSHIM_SPECFRAME_H

#include <memory>
#include <vector>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Frame.h"

namespace ast {

/**
A specialised form of one-dimensional Frame which represents various coordinate systems
used to describe positions within an electro-magnetic spectrum.

The particular coordinate system to be
used is specified by setting the SpecFrame_System "System" attribute (the
default is wavelength) qualified, as necessary, by other attributes
such as the rest frequency, the standard of rest, the epoch of
observation, units, etc (see `Attributes` for details).

By setting a value for the SpecFrame_SpecOrigin "SpecOrigin" attribute,
a @ref SpecFrame can be made to represent offsets from a given spectral position,
rather than absolute spectral values.

### Attributes

In addition to those attributes common to all Frames, every
@ref SpecFrame also has the following attributes:

- @ref SpecFrame_AlignSpecOffset "AlignSpecOffset":
    align @ref SpecFrame "SpecFrames" using the offset coordinate system?
- @ref SpecFrame_AlignStdOfRest "AlignStdOfRest":
    standard of rest in which to align @ref SpecFrame "SpecFrames".
- @ref SpecFrame_RefDec "RefDec": declination of the source (FK5 J2000, "dd:mm:ss.s").
- @ref SpecFrame_RefRA "RefRA": right ascension of the source (FK5 J2000, "hh:mm:ss.s").
- @ref SpecFrame_RestFreq "RestFreq": rest frequency.
- @ref SpecFrame_SourceSys "SourceSys": source velocity spectral system.
- @ref SpecFrame_SourceVel "SourceVel": source velocity.
- @ref SpecFrame_SourceVRF "SourceVRF": source velocity rest frame.
- @ref SpecFrame_SpecOrigin "SpecOrigin": the zero point for @ref SpecFrame axis values.
- @ref SpecFrame_StdOfRest "StdOfRest": standard of rest.

Several of the Frame attributes inherited by the @ref SpecFrame class
refer to a specific axis of the @ref Frame (for instance @ref Frame_Unit "Unit(axis)",
@ref Frame_Label "Label(axis)", etc). Since a @ref SpecFrame is strictly one-dimensional,
it allows these attributes to be specified without an axis index.
So for instance, "Unit" is allowed in place of "Unit(1)".

### Examples

- `frame = SpecFrame()`

      Create a @ref SpecFrame to describe the default wavelength spectral
      coordinate system. The SpecFrame_RestFreq "RestFreq" attribute (rest frequency) is
      unspecified, so it will not be possible to align this @ref SpecFrame
      with another @ref SpecFrame on the basis of a velocity-based system. The
      standard of rest is also unspecified. This means that alignment
      will be possible with other @ref SpecFrame "SpecFrames", but no correction will be
      made for Doppler shift caused by change of rest frame during the
      alignment.

- `frame = SpecFrame("System=VELO, RestFreq=1.0E15, StdOfRest=LSRK")`

      Create a @ref SpecFrame describing a apparent radial velocity ("VELO") axis
      with rest frequency 1.0E15 Hz (about 3000 Angstroms), measured
      in the kinematic Local Standard of Rest ("LSRK"). Since the
      source position has not been specified (using attributes @ref SpecFrame_RefRA "RefRA" and
      @ref SpecFrame_RefDec "RefDec"), it will only be possible to align this @ref SpecFrame with
      other @ref SpecFrame "SpecFrames" which are also measured in the LSRK standard of
      rest.

### Notes

- When conversion between two @ref SpecFrame "SpecFrames" is requested (as when
supplying @ref SpecFrame "SpecFrames" to @ref Frame.convert)
account will be taken of the nature of the spectral coordinate systems
they represent, together with any qualifying rest frequency, standard
of rest, epoch values, etc. The @ref Frame_AlignSystem "AlignSystem"
and SpecFrame_AlignStdOfRest "AlignStdOfRest" attributes will also be taken into account.
The results will therefore fully reflect the relationship between positions measured in the two
systems. In addition, any difference in the Unit attributes of the two
systems will also be taken into account.
*/
class SpecFrame : public Frame {
    friend class Object;

public:
    /**
    Construct a @ref SpecFrame

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit SpecFrame(std::string const &options = "")
            : Frame(reinterpret_cast<AstFrame *>(astSpecFrame("%s", options.c_str()))) {}

    virtual ~SpecFrame() {}

    /// Copy constructor: make a deep copy
    SpecFrame(SpecFrame const &) = default;
    SpecFrame(SpecFrame &&) = default;
    SpecFrame &operator=(SpecFrame const &) = delete;
    SpecFrame &operator=(SpecFrame &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SpecFrame> copy() const {
        return std::static_pointer_cast<SpecFrame>(copyPolymorphic());
    }

    /**
    Get @ref SpecFrame_AlignSpecOffset "AlignSpecOffset":
    align @ref SpecFrame "SpecFrames" using the offset coordinate system?
    */
    bool getAlignSpecOffset() const { return getB("AlignSpecOffset"); }

    /**
    Get @ref SpecFrame_AlignStdOfRest "AlignStdOfRest":
    standard of rest in which to align @ref SpecFrame "SpecFrames".
    */
    std::string getAlignStdOfRest() const { return getC("AlignStdOfRest"); }

    /// Get @ref SpecFrame_RefDec "RefDec": declination of the source (FK5 J2000, "dd:mm:ss.s").
    std::string getRefDec() const { return getC("RefDec"); }

    /// Get @ref SpecFrame_RefRA "RefRA": right ascension of the source (FK5 J2000, "hh:mm:ss.s").
    std::string getRefRA() const { return getC("RefRA"); }

    /**
    Get the reference position (specified by @ref SpecFrame_RefRA "RefRA"
    and @ref SpecFrame_RefDec "RefDec") converted to
    the celestial coordinate system represented by a supplied @ref SkyFrame.

    @param[in] frm  @ref SkyFrame which defines the required celestial coordinate system.
    @return the reference longitude and latitude in
        the coordinate system represented by the supplied @ref SkyFrame (radians).
    */
    std::vector<double> getRefPos(SkyFrame const &frm) const {
        std::vector<double> ret(2);
        astGetRefPos(getRawPtr(), frm.getRawPtr(), &ret[0], &ret[1]);
        assertOK();
        detail::astBadToNan(ret);
        return ret;
    }

    /**
    Return the reference position (specified by @ref SpecFrame_RefRA "RefRA"
    and @ref SpecFrame_RefDec "RefDec") as FK5 J2000 RA, Dec (radians).
    */
    std::vector<double> getRefPos() const {
        std::vector<double> ret(2);
        // using NULL instead of nullptr avoids a compiler warning about comparing NULL to nullptr_t
        astGetRefPos(getRawPtr(), NULL, &ret[0], &ret[1]);
        assertOK();
        detail::astBadToNan(ret);
        return ret;
    }

    /// Get @ref SpecFrame_RestFreq "RestFreq": rest frequency (GHz).
    double getRestFreq() const { return getD("RestFreq"); }

    /// Get @ref SpecFrame_SourceSys "SourceSys": source velocity spectral system.
    std::string getSourceSys() const { return getC("SourceSys"); }

    /**
    Get @ref SpecFrame_SourceVel "SourceVel": source velocity
    (in the system specified by @ref SpecFrame_SourceSys "SourceSys").
    */
    double getSourceVel() const { return getD("SourceVel"); }

    /// Get @ref SpecFrame_SourceVRF "SourceVRF": source velocity rest frame.
    std::string getSourceVRF() const { return getC("SourceVRF"); }

    /// Get @ref SpecFrame_SpecOrigin "SpecOrigin": the zero point for @ref SpecFrame axis values.
    double getSpecOrigin() const { return getD("SpecOrigin"); }

    /// Get @ref SpecFrame_StdOfRest "StdOfRest": standard of rest.
    std::string getStdOfRest() const { return getC("StdOfRest"); }

    /**
    Set @ref SpecFrame_AlignSpecOffset "AlignSpecOffset":
    align @ref SpecFrame "SpecFrames" using the offset coordinate system?
    */
    void setAlignSpecOffset(bool align) { setB("AlignSpecOffset", align); }

    /**
    Set @ref SpecFrame_AlignStdOfRest "AlignStdOfRest":
    standard of rest in which to align @ref SpecFrame "SpecFrames".
    */
    void setAlignStdOfRest(std::string const &stdOfRest) { setC("AlignStdOfRest", stdOfRest); }

    /// Set @ref SpecFrame_RefDec "RefDec": declination of the source (FK5 J2000, "dd:mm:ss.s").
    void setRefDec(std::string const &refDec) { setC("RefDec", refDec); }

    /// Set @ref SpecFrame_RefRA "RefRA": right ascension of the source (FK5 J2000, "hh:mm:ss.s").
    void setRefRA(std::string const &refRA) { setC("RefRA", refRA); }

    /**
    Set the reference position (@ref SpecFrame_RefRA "RefRA" and @ref SpecFrame_RefDec "RefDec")
    using axis values (in radians) supplied within the celestial coordinate system
    represented by a supplied @ref SkyFrame.

    @param[in] frm  @ref SkyFrame which defines the celestial coordinate system in which
            the longitude and latitude values are supplied.
    @param[in] lon  The longitude of the reference point, in the coordinate system
            represented by the supplied @ref SkyFrame (radians).
    @param[in] lat  The latitude of the reference point, in the coordinate system
            represented by the supplied @ref SkyFrame (radians).
    */
    void setRefPos(SkyFrame const &frm, double lon, double lat) {
        astSetRefPos(getRawPtr(), frm.getRawPtr(), lon, lat);
        assertOK();
    }

    /**
    Set the reference position (@ref SpecFrame_RefRA "RefRA" and @ref SpecFrame_RefDec "RefDec")
    from FK5 J2000 RA and Dec (radians).

    @param[in] ra  FK5 J2000 RA (radians).
    @param[in] dec  FK5 J2000 Dec (radians).
    */
    void setRefPos(double ra, double dec) {
        // using NULL instead of nullptr avoids a compiler warning about comparing NULL to nullptr_t
        astSetRefPos(getRawPtr(), NULL, ra, dec);
        assertOK();
    }

    /// Set @ref SpecFrame_RestFreq "RestFreq": rest frequency in GHz.
    void setRestFreq(double freq) { setD("RestFreq", freq); }

    /// Set @ref SpecFrame_RestFreq "RestFreq": rest frequency in user-specified units.
    void setRestFreq(std::string const &freq) { setC("RestFreq", freq); }

    /// Set @ref SpecFrame_SourceSys "SourceSys": source velocity spectral system.
    void setSourceSys(std::string const &system) { setC("SourceSys", system); }

    /**
    Set @ref SpecFrame_SourceVel "SourceVel": source velocity
    (in the system specified by @ref SpecFrame_SourceSys "SourceSys").
    */
    void setSourceVel(double vel) { setD("SourceVel", vel); }

    /// Set @ref SpecFrame_SourceVRF "SourceVRF": source velocity @ref SpecFrame_StandardsOfRest "rest frame".
    void setSourceVRF(std::string const &vrf) { setC("SourceVRF", vrf); }

    /// Set @ref SpecFrame_SpecOrigin "SpecOrigin": the zero point for @ref SpecFrame axis values.
    void setSpecOrigin(double origin) { setD("SpecOrigin", origin); }

    /// Set @ref SpecFrame_StdOfRest "StdOfRest": @ref SpecFrame_StandardsOfRest "standard of rest".
    void setStdOfRest(std::string const &stdOfRest) { setC("StdOfRest", stdOfRest); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SpecFrame, AstSpecFrame>();
    }

    /// Construct a SpecFrame from a raw AST pointer
    explicit SpecFrame(AstSpecFrame *rawptr) : Frame(reinterpret_cast<AstFrame *>(rawptr)) {
        if (!astIsASpecFrame(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a SpecFrame";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
