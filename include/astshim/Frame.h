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
#ifndef ASTSHIM_FRAME_H
#define ASTSHIM_FRAME_H

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "astshim/Mapping.h"
#include "astshim/Object.h"
#include "astshim/base.h"
#include "astshim/detail/utils.h"

namespace ast {

class CmpFrame;
class Frame;

/**
Struct returned by @ref Frame.offset2 containing a direction and a point
*/
class DirectionPoint {
public:
    /**
    Construct a DirectionPoint

    @param[in] direction  Direction, and angle in radians
    @param[in] point  Point
    */
    DirectionPoint(double direction, PointD const &point) : direction(direction), point(point){};
    double direction;  ///< Direction, an angle in radians
    PointD point;      ///< Point
};

/**
Struct returned by @ref Frame.unformat containing the number of characters read and corresponding value
*/
class NReadValue {
public:
    /**
    Construct an NReadValue

    @param[in] nread  Number of characters that was read
    @param[in] value  Value that was read
    */
    NReadValue(int nread, double value) : nread(nread), value(value){};
    int nread;     ///< Number of characters that was read
    double value;  ///< Value that was read
};

/**
Struct returned by @ref Frame.resolve containing a point and the resolved vector components
*/
class ResolvedPoint {
public:
    /**
    Construct an empty ResolvedPoint

    @param[in] naxes  Number of axes in the point
    */
    explicit ResolvedPoint(int naxes) : point(naxes), d1(), d2() {}
    std::vector<double> point;  ///< Point
    double d1;                  ///< Resolved vector component 1
    double d2;                  ///< Resolved vector component 2
};

/**
Struct returned by @ref Frame.pickAxes containing a frame and a mapping
*/
class FrameMapping {
public:
    /**
    Construct a FrameMapping

    @param[in,out] frame  Frame
    @param[in,out] mapping  Mapping
    */
    FrameMapping(std::shared_ptr<Frame> frame, std::shared_ptr<Mapping> mapping)
            : frame(frame), mapping(mapping) {}
    std::shared_ptr<Frame> frame;      ///< Frame
    std::shared_ptr<Mapping> mapping;  ///< Mapping
};

class FrameSet;

/**
Frame is used to represent a coordinate system.

It does this in rather the same way that a frame around a graph describes
the coordinate space in which data are plotted. Consequently, a Frame has
a Title (string) attribute, which describes the coordinate space,
and contains axes which in turn hold information such as Label and Units strings
which are used for labelling (e.g.) graphical output. In general, however,
the number of axes is not restricted to two.

Functions are available for converting Frame coordinate values into a form
suitable for display, and also for calculating distances and offsets
between positions within the Frame.

Frames may also contain knowledge of how to transform to and from related coordinate systems.

### Attributes

In addition to those provided by @ref Mapping and @ref Object,
Frame provides the following attributes, where `axis` is
an axis number, starting from 1 and `(axis)` may be omitted
if the Frame has only one axis:
- @ref Frame_ActiveUnit "ActiveUnit": pay attention to units when one @ref Frame
    is used to match another? (Note: not a true attribute, but close).
- @ref Frame_AlignSystem "AlignSystem": Coordinate system used to align Frames
- @ref Frame_Bottom "Bottom(axis)": Lowest axis value to display
- @ref Frame_Digits "Digits/Digits(axis)`: Number of digits of precision
- @ref Frame_Direction "Direction(axis)": Display axis in conventional direction?
- @ref Frame_Domain "Domain": Coordinate system domain
- @ref Frame_Dut1 "Dut1": Difference between the UT1 and UTC timescale (sec)
- @ref Frame_Epoch "Epoch": Epoch of observation
- @ref Frame_Format "Format(axis)": Format specification for axis values
- @ref Frame_InternalUnit "InternalUnit(axis)": Physical units for unformated axis values
- @ref Frame_Label "Label(axis)": Axis label
- @ref Frame_MatchEnd "MatchEnd": Match trailing axes?
- @ref Frame_MaxAxes "MaxAxes": Maximum number of axes a frame found by @ref findFrame may have.
- @ref Frame_MinAxes "MinAxes": Minimum number of axes a frame found by @ref findFrame may have.
- @ref Frame_NAxes "NAxes": Number of Frame axes
- @ref Frame_NormUnit "NormUnit(axis)": Normalised physical units for formatted axis values
- @ref Frame_ObsAlt "ObsAlt": Geodetic altitude of observer (m)
- @ref Frame_ObsLat "ObsLat": Geodetic latitude of observer
- @ref Frame_ObsLon "ObsLon": Geodetic longitude of observer
- @ref Frame_Permute "Permute": Allow axis permutation when used as a template?
- @ref Frame_PreserveAxes "PreserveAxes": Preserve axes?
- @ref Frame_Symbol "Symbol(axis)": Axis symbol
- @ref Frame_System "System": Coordinate system used to describe positions within the domain
- @ref Frame_Title "Title: Frame title
- @ref Frame_Top "Top(axis)": Highest axis value to display
- @ref Frame_Unit "Unit(axis)": Physical units for formatted axis values
*/
class Frame : public Mapping {
    friend class Object;

public:
    /**
    Construct a Frame

    @param[in] naxes  The number of Frame axes (i.e. the number of dimensions
                    of the coordinate space which the Frame describes).
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit Frame(int naxes, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astFrame(naxes, "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~Frame() {}

    /// Copy constructor: make a deep copy
    Frame(Frame const &) = default;
    Frame(Frame &&) = default;
    Frame &operator=(Frame const &) = delete;
    Frame &operator=(Frame &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<Frame> copy() const { return std::static_pointer_cast<Frame>(copyPolymorphic()); }

    /**
    Find the angle at point B between the line joining points A and B, and the line joining points C and B.

    These lines will in fact be geodesic curves appropriate to the Frame in use. For instance, in SkyFrame,
    they will be great circles.

    @param[in] a  the coordinates of the first point
    @param[in] b  the coordinates of the first second
    @param[in] c  the coordinates of the first third

    @return The angle in radians, from the line AB to the line CB. If the Frame is 2-dimensional,
        it will be in the range [-pi, pi], and positive rotation is in the same sense as rotation
        from the positive direction of axis 2 to the positive direction of axis 1. If the
        Frame has more than 2 axes, a positive value will always be returned in the range (0, pi].

    @throws std::invalid_argument if `a`, `b` or `c` have the wrong length
    */
    double angle(PointD const &a, PointD const &b, PointD const &c) const {
        assertPointLength(a, "a");
        assertPointLength(b, "b");
        assertPointLength(c, "c");
        return detail::safeDouble(astAngle(getRawPtr(), a.data(), b.data(), c.data()));
    }

    /**
    Find the angle, as seen from point A, between the positive direction of a specified axis,
    and the geodesic curve joining point A to point B.

    @param[in] a  the coordinates of the first point
    @param[in] b  the coordinates of the second point
    @param[in] axis  the index of the axis from which the angle is to be measured, where 1 is the first axis

    @return The angle in radians, from the positive direction of the specified axis, to the line AB.
        If the Frame is 2-dimensional, it will be in the range [-pi/2, +pi/2],
        and positive rotation is in the same sense as rotation from the positive direction of axis 2 to
        the positive direction of axis 1. If the Frame has more than 2 axes, a positive value
        will always be returned in the range (0, pi]

    @throws std::invalid_argument if `a` or `b` have the wrong length

    ### Notes:

    - The geodesic curve used by this function is the path of shortest distance between two points,
        as defined by the `distance` method.
    */
    double axAngle(PointD const &a, PointD const &b, int axis) const {
        assertPointLength(a, "a");
        assertPointLength(b, "b");
        return detail::safeDouble(astAxAngle(getRawPtr(), a.data(), b.data(), axis));
    }

    /**
    Return a signed value representing the axis increment from axis value v1 to axis value v2.

    For a simple Frame, this is a trivial operation returning the difference between the two axis values.
    But for other derived classes of Frame (such as a SkyFrame) this is not the case.

    @param[in] axis  The index of the axis to which the supplied values refer. The first axis has index 1.
    @param[in] v1  The first axis value.
    @param[in] v2  The second axis value.

    @return The distance from the first to the second axis value.
    */
    double axDistance(int axis, double v1, double v2) const {
        return detail::safeDouble(astAxDistance(getRawPtr(), axis, v1, v2));
    }

    /**
    Return an axis value formed by adding a signed axis increment onto a supplied axis value.

    For a simple Frame, this is a trivial operation returning the sum of the two supplied values.
    But for other derived classes of Frame (such as a SkyFrame) this is not the case.

    @param[in] axis  The index of the axis to which the supplied values refer. The first axis has index 1.
    @param[in] v1  The original axis value.
    @param[in] dist  The axis increment to add to the original axis value.

    @return The incremented axis value
    */
    double axOffset(int axis, double v1, double dist) const {
        return detail::safeDouble(astAxOffset(getRawPtr(), axis, v1, dist));
    }

    /**
    Compute a frameset that describes the conversion between this frame and another frame.

    If conversion is possible, it returns a shared pointer to a @ref FrameSet which describes
    the conversion and which may be used (as a Mapping) to transform coordinate values in either direction.
    Otherwise it returns an empty shared pointer.

    The same function may also be used to determine how to convert between two @ref FrameSet "FrameSets"
    (or between a Frame and a @ref FrameSet, or vice versa). This mode is intended for use when
    (for example) two images have been calibrated by attaching a @ref FrameSet to each.
    `convert` might then be used to search for a celestial coordinate system that both
    images have in common, and the result could then be used to convert between the pixel coordinates
    of both images – having effectively used their celestial coordinate systems to align them.

    When using @ref FrameSet "FrameSets", there may be more than one possible intermediate coordinate system
    in which to perform the conversion (for instance, two @ref FrameSet "FrameSets" might both have
    celestial coordinates, detector coordinates, pixel coordinates, etc.).
    A comma-separated list of coordinate system domains may therefore be given
    which defines a priority order to use when selecting the intermediate coordinate system.
    The path used for conversion must go via an intermediate coordinate system
    whose Domain attribute matches one of the domains given. If conversion cannot be achieved
    using the first domain, the next one is considered, and so on, until success is achieved.

    ### Applicability

    - @ref DSBSpecFrame

        If the AlignSideBand attribute is non-zero, alignment occurs in the
        upper sideband expressed within the spectral system and standard of
        rest given by attributes @ref Frame_AlignSystem "AlignSystem" and `AlignStdOfRest`. If
        `AlignSideBand` is zero, the two DSBSpecFrames are aligned as if
        they were simple SpecFrames (i.e. the SideBand is ignored).

    - @ref Frame

        This function applies to all Frames. Alignment occurs within the
        coordinate system given by attribute AlignSystem.

    - @ref FrameSet

        If either this object or `to` is a @ref FrameSet,
        then this method will attempt to convert from the
        coordinate system described by the current Frame of the "from"
        @ref FrameSet to that described by the current Frame of the "to"
        @ref FrameSet.

        To achieve this, it will consider all of the Frames within
        each @ref FrameSet as a possible way of reaching an intermediate
        coordinate system that can be used for the conversion. There
        is then the possibility that more than one conversion path
        may exist and, unless the choice is sufficiently restricted
        by the "domainlist" string, the sequence in which the Frames
        are considered can be important. In this case, the search
        for a conversion path proceeds as follows:
        - Each field in the "domainlist" string is considered in turn.
        - The Frames within each @ref FrameSet are considered in a
        specific order: (1) the base Frame is always considered
        first, (2) after this come all the other Frames in
        Frame-index order (but omitting the base and current Frames),
        (3) the current Frame is always considered last.  However, if
        either @ref FrameSet's Invert attribute is set to a non-zero value
        (so that the @ref FrameSet is inverted), then its Frames are
        considered in reverse order. (Note that this still means that
        the base Frame is considered first and the current Frame
        last, because the Invert value will also cause these Frames
        to swap places.)
        - All source Frames are first considered (in the appropriate
        order) for conversion to the first destination Frame. If no
        suitable intermediate coordinate system emerges, they are
        then considered again for conversion to the second
        destination Frame (in the appropriate order), and so on.
        - Generally, the first suitable intermediate coordinate
        system found is used. However, the overall Mapping between
        the source and destination coordinate systems is also
        examined.  Preference is given to cases where both the
        forward and inverse transformations are defined. If only one
        transformation is defined, the forward one is preferred.
        - If the domain of the intermediate coordinate system matches
        the current "domainlist" field, the conversion path is
        accepted. Otherwise, the next "domainlist" field is considered
        and the process repeated.

        If conversion is possible, the Base attributes of the two
        @ref FrameSet "FrameSets" will be modified on exit to identify the Frames
        used to access the intermediate coordinate system which was
        finally accepted.

        Note that it is possible to force a particular Frame within a
        @ref FrameSet to be used as the basis for the intermediate
        coordinate system, if it is suitable, by (a) focussing
        attention on
        it by specifying its domain in the "domainlist" string, or (b)
        making it the base Frame, since this is always considered
        first.

    - @ref SpecFrame

        Alignment occurs within the spectral system and standard of rest
        given by attributes @ref Frame_AlignSystem "AlignSystem" and `AlignStdOfRest`.

    - @ref TimeFrame

        Alignment occurs within the time system and time scale given by
        attributes @ref Frame_AlignSystem "AlignSystem" and `AlignTimeScale`.

    ### Examples

    - `auto cvt = a.convert(b)`

        Obtain a @ref FrameSet that converts between the coordinate systems
        represented by "a" and "b" (assumed to be Frames).

    - `auto cvt = SkyFrame().convert(SkyFrame("Equinox=2005"))`

        Create a @ref FrameSet which describes precession in the default
        FK5 celestial coordinate system between equinoxes J2000 (also
        the default) and J2005. The returned "cvt" @ref FrameSet may be used
        to apply this precession correction to
        any number of coordinate values given in radians.

        Note that the returned @ref FrameSet also contains information
        about how to format coordinate values. This means that
        setting its Report attribute to 1 is a simple way to obtain
        printed output (formatted in sexagesimal notation) to show
        the coordinate values before and after conversion.

    - `auto cvt = a.convert(b, "sky,detector,")`

        Create a @ref FrameSet that converts between the coordinate systems
        represented by the current Frames of "a" and "b"
        (now assumed to be @ref FrameSet "FrameSets"), via the intermediate "SKY"
        coordinate system.  This, by default, is the Domain
        associated with a celestial coordinate system represented by
        a SkyFrame.

        If this fails (for example, because either @ref FrameSet lacks
        celestial coordinate information), then the user-defined
        "DETECTOR" coordinate system is used instead. If this also
        fails, then all other possible ways of achieving conversion
        are considered before giving up.

        The returned "cvt" @ref FrameSet describes the conversion.

        The Base attributes of the two @ref FrameSet
        will be set by `ref convert to indicate which of their Frames was
        used for the intermediate coordinate system. This means that
        you can subsequently determine which coordinate system was
        used by enquiring the Domain attribute of either base Frame.

    @param[in,out] to  A Frame which represents the "destination" coordinate system.
                This is the coordinate system into which you wish to convert your coordinates.
                If a @ref FrameSet is given, its current Frame (as determined by its Current attribute)
                is taken to describe the destination coordinate system. Note that the Base attribute
                of this @ref FrameSet may be modified by this function to indicate which intermediate
                coordinate system was used.
    @param[in] domainlist  A string containing a comma-separated list of Frame domains.
                This may be used to define a priority order for the different
                intermediate coordinate systems that might be used to perform the conversion.
                The function will first try to obtain a conversion by making use only of an intermediate
                coordinate system whose Domain attribute matches the first domain in this list.
                If this fails, the second domain in the list will be used, and so on, until conversion
                is achieved.  A blank domain (e.g.  two consecutive commas) indicates that all
                coordinate systems should be considered, regardless of their domains.
    @return A @ref FrameSet which describes the conversion and contains two Frames,
        or an empty shared pointer if the conversion is not possible.
        Frame number 1 (its base Frame) will describe the source coordinate
        system, corresponding to the "from" parameter. Frame number 2
        (its current Frame) will describe the destination coordinate
        system, corresponding to the "to" parameter. The Mapping
        which inter-relates these two Frames will perform the
        required conversion between their respective coordinate
        systems.
        Note that a @ref FrameSet may be used both as a @ref Mapping and as a
        @ref Frame. If the result is used as a @ref Mapping
        then it provides a means of converting coordinates
        from the source to the destination coordinate system (or
        vice versa if its inverse transformation is selected). If it
        is used as a Frame, its attributes will describe the
        destination coordinate system.

    ### Notes

    -  The Mapping represented by the returned @ref FrameSet results in
        alignment taking place in the coordinate system specified by the
        AlignSystem attribute of the "to" Frame. See the description of the
        AlignSystem attribute for further details.
    - When aligning (say) two images, which have been calibrated by
        attaching @ref FrameSet "FrameSets" to them, it is usually necessary to convert
        between the base @ref Frame "Frames" (representing "native" pixel coordinates)
        of both @ref FrameSet "FrameSets". This may be achieved by obtaining the inverses
        of the @ref FrameSet "FrameSets" (using @ref Mapping.inverted "inverted").
    */
    std::shared_ptr<FrameSet> convert(Frame const &to, std::string const &domainlist = "");

    /**
    Find the distance between two points whose Frame coordinates are given.

    The distance calculated is that along the geodesic curve that joins the two points.
    For example, in a basic Frame, the distance calculated will be the Cartesian distance along the straight
    line joining the two points. For a more specialised Frame describing a sky coordinate system, however,
    it would be the distance along the great circle passing through two sky positions.

    @param[in] point1  The coordinates of the first point.
    @param[in] point2  The coordinates of the second point.

    @return The distance between the two points.
    */
    double distance(PointD const &point1, PointD const &point2) const {
        assertPointLength(point1, "point1");
        assertPointLength(point2, "point2");
        return detail::safeDouble(astDistance(getRawPtr(), point1.data(), point2.data()));
    }

    /**
    Find a coordinate system with specified characteristics.

    Use a "template" Frame to search another Frame
    (or @ref FrameSet) to identify a coordinate system which has a
    specified set of characteristics. If a suitable coordinate
    system can be found, the function returns a pointer to a
    @ref FrameSet which describes the required coordinate system and how
    to convert coordinates to and from it.

    This function is provided to help answer general questions about
    coordinate systems, such as typically arise when coordinate
    information is imported into a program as part of an initially
    unknown dataset. For example:
    - Is there a wavelength scale?
    - Is there a 2-dimensional coordinate system?
    - Is there a celestial coordinate system?
    - Can I plot the data in ecliptic coordinates?

    You can also use this function as a means of reconciling a
    user's preference for a particular coordinate system (for
    example, what type of axes to draw) with what is actually
    possible given the coordinate information available.

    To perform a search, you supply a "target" Frame (or @ref FrameSet)
    which represents the set of coordinate systems to be searched.
    If a basic Frame is given as the target, this set of coordinate
    systems consists of the one described by this Frame, plus all
    other "virtual" coordinate systems which can potentially be
    reached from it by applying built-in conversions (for example,
    any of the celestial coordinate conversions known to the AST
    library would constitute a "built-in" conversion). If a @ref FrameSet
    is given as the target, the set of coordinate systems to be
    searched consists of the union of those represented by all the
    individual Frames within it.

    To select from this large set of possible coordinate systems,
    you supply a "template" Frame which is an instance of the type
    of Frame you are looking for. Effectively, you then ask the
    function to "find a coordinate system that looks like this".

    You can make your request more or less specific by setting
    attribute values for the template Frame. If a particular
    attribute is set in the template, then the function will only
    find coordinate systems which have exactly the same value for
    that attribute.  If you leave a template attribute un-set,
    however, then the function has discretion about the value the
    attribute should have in any coordinate system it finds. The
    attribute will then take its value from one of the actual
    (rather than virtual) coordinate systems in the target. If the
    target is a @ref FrameSet, its Current attribute will be modified to
    indicate which of its Frames was used for this purpose.

    The result of this process is a coordinate system represented by
    a hybrid Frame which acquires some attributes from the template
    (but only if they were set) and the remainder from the
    target. This represents the "best compromise" between what you
    asked for and what was available. A Mapping is then generated
    which converts from the target coordinate system to this hybrid
    one, and the returned @ref FrameSet encapsulates all of this
    information.

    ### Applicability to Subclasses

    - @ref FrameSet

        If the target is a @ref FrameSet, the possibility exists that
        several of the Frames within it might be matched by the
        template.  Unless the choice is sufficiently restricted by
        the "domainlist" string, the sequence in which Frames are
        searched can then become important. In this case, the search
        proceeds as follows:
        - Each field in the "domainlist" string is considered in turn.
        - An attempt is made to match the template to each of the
            target's Frames in the order:
            (1) the current Frame
            (2) the base Frame
            (3) each remaining Frame in the order of being added to the target @ref FrameSet
        - Generally, the first match found is used. However, the
            Mapping between the target coordinate system and the
            resulting Frame is also examined. Preference is given to
            cases where both the forward and inverse transformations are
            defined. If only one transformation is defined, the
            forward one is preferred.
        - If a match is found and the domain of the resulting Frame also
            matches the current "domainlist" field, it is
            accepted. Otherwise, the next "domainlist" field is considered
            and the process repeated.

        If a suitable coordinate system is found, then the @ref FrameSet's Current attribute
        will be modified to indicate which Frame was used to obtain attribute values
        which were not specified by the template.  This Frame will, in some sense,
        represent the "closest" non-virtual coordinate system to the one you requested.

    ### Examples

    - `auto result = target.findFrame(ast::Frame(3))`

        Search for a 3-dimensional coordinate system in the target
        Frame (or @ref FrameSet). No attributes have been set in the
        template Frame (created by ast::Frame), so no restriction has
        been placed on the required coordinate system, other than
        that it should have 3 dimensions. The first suitable Frame
        found will be returned as part of the "result" @ref FrameSet.

    - `auto result = target.findFrame(astSkyFrame())`

        Search for a celestial coordinate system in the target
        Frame (or @ref FrameSet). The type of celestial coordinate system
        is unspecified, so astFindFrame will return the first one
        found as part of the "result" @ref FrameSet. If the target is
        a @ref FrameSet, then its Current attribute will be updated to
        identify the Frame that was used.

    - `auto result = target.findFrame(astSkyFrame("MaxAxes=100"))`

        This is like the last example, except that in the event of the
        target being a CmpFrame, the component Frames encapsulated by the
        CmpFrame will be searched for a SkyFrame. If found, the returned
        Mapping will included a PermMap which selects the required axes
        from the target CmpFrame.

        This is acomplished by setting the MaxAxes attribute of the
        template SkyFrame to a large number (larger than or equal to the
        number of axes in the target CmpFrame). This allows the SkyFrame
        to be used as a match for Frames containing from 2 to 100 axes.

    - `auto result = target.findFrame(astSkyFrame("System=FK5"))`

        Search for an equatorial (FK5) coordinate system in the
        target. The `Equinox` value for the coordinate system has not
        been specified, so will be obtained from the target. If the
        target is a @ref FrameSet, its `Current` attribute will be updated
        to indicate which @ref SkyFrame was used to obtain this value.

    - `auto result = target.findFrame(astFrame(2), "sky,pixel,")`

        Search for a 2-dimensional coordinate system in the target.
        Initially, a search is made for a suitable coordinate system
        whose Domain attribute has the value "SKY". If this search fails,
        a search is then made for one with the domain "PIXEL".
        If this also fails, then any 2-dimensional coordinate system
        is returned as part of the "result" @ref FrameSet.

        Only if no 2-dimensional coordinate systems can be reached by
        applying built-in conversions to any of the Frames in the
        target will the search fail.

    - `auto result = target.findFrame(astFrame(1, "Domain=WAVELENGTH"))`

        Searches for any 1-dimensional coordinate system in the
        target which has the domain "WAVELENGTH".

    - `auto result = target.findFrame(astFrame(1), "wavelength")`

        This example has exactly the same effect as that above. It
        illustrates the equivalence of the template's Domain attribute
        and the fields in the "domainlist" string.

    - `auto result = target.findFrame(Frame(1, "MaxAxes=3"))`

        This is a more advanced example which will search for any
        coordinate system in the target having 1, 2 or 3
        dimensions. The Frame returned (as part of the "result"
        @ref FrameSet) will always be 1-dimensional, but will be related
        to the coordinate system that was found by a suitable Mapping
        (e.g. a PermMap) which simply extracts the first axis.

        If we had wanted a Frame representing the actual (1, 2 or
        3-dimensional) coordinate system found, we could set the
        PreserveAxes attribute to a non-zero value in the template.

    - `auto result = target.findFrame(SkyFrame("Permute=0"))`

        Search for any celestial coordinate system in the target,
        but only finds one if its axes are in the conventional
        (longitude,latitude) order and have not been permuted
        (e.g. with astPermAxes).

    ### More on Using Templates

    A Frame (describing a coordinate system) will be found by this
    function if (a) it is "matched" by the template you supply, and
    (b) the value of its Domain attribute appears in the "domainlist"
    string (except that a blank field in this string permits any
    domain). A successful match by the template depends on a number
    of criteria, as outlined below:
    - In general, a template will only match another Frame which
    belongs to the same class as the template, or to a derived (more
    specialised) class. For example, a SkyFrame template will match
    any other SkyFrame, but will not match a basic
    Frame. Conversely, a basic Frame template will match any class
    of Frame.
    - The exception to this is that a Frame of any class can be used to
    match a CmpFrame, if that CmpFrame contains a Frame of the same
    class as the template. Note however, the MaxAxes and MinAxes
    attributes of the template must be set to suitable values to allow
    it to match the CmpFrame. That is, the MinAxes attribute must be
    less than or equal to the number of axes in the target, and the MaxAxes
    attribute must be greater than or equal to the number of axes in
    the target.
    - If using a CmpFrame as a template frame, the MinAxes and MaxAxes
    for the template are determined by the MinAxes and MaxAxes values of
    the component Frames within the template. So if you want a template
    CmpFrame to be able to match Frames with different numbers of axes,
    then you must set the MaxAxes and/or MinAxes attributes in the component
    template Frames, before combining them together into the template
    CmpFrame.
    - If a template has a value set for any of its main attributes, then
    it will only match Frames which have an identical value for that
    attribute (or which can be transformed, using a built-in
    conversion, so that they have the required value for that
    attribute). If any attribute in the template is un-set, however,
    then Frames are matched regardless of the value they may have
    for that attribute. You may therefore make a template more or
    less specific by choosing the attributes for which you set
    values. This requirement does not apply to 'descriptive' attributes
    such as titles, labels, symbols, etc.
    - An important application of this principle involves the Domain
    attribute. Setting the Domain attribute of the template has the
    effect of restricting the search to a particular type of Frame
    (with the domain you specify).  Conversely, if the Domain
    attribute is not set in the template, then the domain of the
    Frame found is not relevant, so all Frames are searched.  Note
    that the
    "domainlist" string provides an alternative way of restricting the
    search in the same manner, but is a more convenient interface if
    you wish to search automatically for another domain if the first
    search fails.
    - Normally, a template will only match a Frame which has the
    same number of axes as itself. However, for some classes of
    template, this default behaviour may be changed by means of the
    MinAxes, MaxAxes and MatchEnd attributes. In addition, the
    behaviour of a template may be influenced by its Permute and
    PreserveAxes attributes, which control whether it matches Frames
    whose axes have been permuted, and whether this permutation is
    retained in the Frame which is returned (as opposed to returning
    the axes in the order specified in the template, which is the
    default behaviour). You should consult the descriptions of these
    attributes for details of this more advanced use of templates.

    @param[in,out] tmplt  Template Frame, which should be an instance of the type of Frame
        you wish to find. If you wanted to find a Frame describing a celestial coordinate
        system, for example, then you might use a SkyFrame here.  See the "Examples"
        section for more ideas.
    @param[in] domainlist  String containing a comma-separated list
        of Frame domains.  This may be used to establish a priority order for the different
        types of coordinate system that might be found.
        The function will first try to find a suitable coordinate system whose Domain
        attribute equals the first domain in this list.  If this fails, the second domain
        in the list will be used, and so on, until a result is obtained.  A blank domain
        (e.g.  two consecutive commas) indicates that any coordinate system is acceptable
        (subject to the template) regardless of its domain.
        This list is case-insensitive and all white space is ignored.  If you do not wish
        to restrict the domain in this way, you should supply an empty string.
    @return A std::shared_ptr<FrameSet> which contains the Frame found and a
        description of how to convert to (and from) the coordinate
        system it represents. If the Frame is not found then return a null pointer.

        This @ref FrameSet will contain two Frames.
        Frame number 1 (its base Frame) represents the target coordinate
        system and will be the same as the (base Frame of the)
        target. Frame number 2 (its current Frame) will be a Frame
        representing the coordinate system which the function
        found. The Mapping which inter-relates these two Frames will
        describe how to convert between their respective coordinate
        systems.
        Note that a @ref FrameSet may be used both as a Mapping and as a
        Frame. If the result is used as a Mapping,
        then it provides a means of converting coordinates
        from the target coordinate system into the new coordinate
        system that was found (and vice versa if its inverse
        transformation is selected). If it is used as a Frame, its
        attributes will describe the new coordinate system.

    ### Notes

    - This method is not const because if called on a @ref FrameSet then the BASE frame
    of the FrameSet may be changed. No other kind of frame will be altered.
    - Similarly the `tmpl` argument is not const because if it is a @ref FrameSet then
    the BASE frame of the template may be changed. No other kind of frame will be altered.
    - The Mapping represented by the returned @ref FrameSet results in
    alignment taking place in the coordinate system specified by the
    @ref Frame_AlignSystem "AlignSystem" attribute of the "template" Frame. See the description
    of the @ref Frame_AlignSystem "AlignSystem" for further details.
    - Beware of setting the `Domain` attribute of the template and then
    using a "domainlist" string which does not include the template's domain
    (or a blank field). If you do so, no coordinate system will be found.
    */
    std::shared_ptr<FrameSet> findFrame(Frame const &tmplt, std::string const &domainlist = "");

    /**
    Return a string containing the formatted (character) version of a coordinate value for a Frame axis.

    The formatting applied is determined by the Frame's attributes and, in particular,
    by any Format attribute string that has been set for the axis.
    A suitable default format (based on the Digits attribute value) will be applied if necessary.

    @param[in] axis  The number of the Frame axis for which formatting is to be performed
                (axis numbering starts at 1 for the first axis).
    @param[in] value   The coordinate value to be formatted.
    */
    std::string format(int axis, double value) const {
        char const *rawstr = astFormat(getRawPtr(), axis, value);
        assertOK();
        return std::string(rawstr);
    }

    /**
    Get @ref Frame_ActiveUnit "ActiveUnit": pay attention to units when one @ref Frame
    is used to match another?
    */
    bool getActiveUnit() const {
        bool ret = astGetActiveUnit(getRawPtr());
        assertOK();
        return ret;
    }

    /**
    Get @ref Frame_AlignSystem "AlignSystem": the coordinate system used
    by @ref convert and @ref findFrame to align Frames
    */
    std::string getAlignSystem() const { return getC("AlignSystem"); }

    /**
    Get @ref Frame_Bottom "Bottom" for one axis: the lowest axis value to display
    */
    double getBottom(int axis) const { return getD(detail::formatAxisAttr("Bottom", axis)); }

    /**
    Get @ref Frame_Digits "Digits": the default used if no specific value specified for an axis
    */
    int getDigits() const { return getI("Digits"); }

    /**
    Get @ref Frame_Digits "Digits" for one axis
    */
    int getDigits(int axis) const { return getI(detail::formatAxisAttr("Digits", axis)); }

    /**
    Get @ref Frame_Direction "Direction" for one axis: display axis in conventional direction?
    */
    bool getDirection(int axis) const { return getB(detail::formatAxisAttr("Direction", axis)); }

    /**
    Get @ref Frame_Domain "Domain": coordinate system domain
    */
    std::string getDomain() const { return getC("Domain"); }

    /**
    Get @ref Frame_Dut1 "Dut1": difference between the UT1 and UTC timescale (sec)
    */
    double getDut1() const { return getD("Dut1"); }

    /**
    Get @ref Frame_Epoch "Epoch": Epoch of observation
    */
    double getEpoch() const { return getD("Epoch"); }

    /**
    Get @ref Frame_Format "Format" for one axis: format specification for axis values.
    */
    std::string getFormat(int axis) const { return getC(detail::formatAxisAttr("Format", axis)); }

    /**
    Get @ref Frame_InternalUnit "InternalUnit(axis)" read-only attribute for one axis:
    physical units for unformated axis values.
    */
    std::string getInternalUnit(int axis) const { return getC(detail::formatAxisAttr("InternalUnit", axis)); }

    /**
    Get @ref Frame_Label "Label(axis)" for one axis: axis label.
    */
    std::string getLabel(int axis) const { return getC(detail::formatAxisAttr("Label", axis)); }

    /**
    Get @ref Frame_MatchEnd "MatchEnd": match trailing axes?
    */
    bool getMatchEnd() const { return getB("MatchEnd"); }

    /**
    Get @ref Frame_MaxAxes "MaxAxes": the maximum axes a frame
    found by @ref findFrame may have.
    */
    int getMaxAxes() const { return getI("MaxAxes"); }

    /**
    Get @ref Frame_MinAxes "MinAxes": the maximum axes a frame
    found by @ref findFrame may have.
    */
    int getMinAxes() const { return getI("MinAxes"); }

    /**
    Get @ref Frame_NAxes "NAxes": the number of axes in the frame
    (i.e. the number of dimensions of the coordinate space which the Frame describes).
    */
    int getNAxes() const { return getI("NAxes"); }

    /**
    Get @ref Frame_NormUnit "NormUnit(axis)" read-only attribute for one frame:
    normalised physical units for formatted axis values
    */
    std::string getNormUnit(int axis) const { return getC(detail::formatAxisAttr("NormUnit", axis)); }

    /**
    Get @ref Frame_ObsAlt "ObsAlt": Geodetic altitude of observer (m).
    */
    double getObsAlt() const { return getD("ObsAlt"); }

    /**
    Get @ref Frame_ObsLat "ObsLat": Geodetic latitude of observer.
    */
    std::string getObsLat() const { return getC("ObsLat"); }

    /**
    Get @ref Frame_ObsLon "ObsLon": Geodetic longitude of observer.
    */
    std::string getObsLon() const { return getC("ObsLon"); }

    /**
    Get @ref Frame_Permute "Permute": allow axis permutation when used as a template?
    */
    bool getPermute() const { return getB("Permute"); }

    /**
    Get @ref Frame_PreserveAxes "PreserveAxes": preserve axes?
    */
    bool getPreserveAxes() const { return getB("PreserveAxes"); }

    /**
    Get @ref Frame_Symbol "Symbol(axis)" for one axis: axis symbol.
    */
    std::string getSymbol(int axis) const { return getC(detail::formatAxisAttr("Symbol", axis)); }

    /**
    Get @ref Frame_System "System": coordinate system used to describe
    positions within the domain.
    */
    std::string getSystem() const { return getC("System"); }

    /**
    Get @ref Frame_Title "Title": frame title.
    */
    std::string getTitle() const { return getC("Title"); }

    /**
    Get @ref Frame_Top "Top": the highest axis value to display
    */
    double getTop(int axis) const { return getD(detail::formatAxisAttr("Top", axis)); }

    /**
    Get @ref Frame_Unit "Unit(axis)" for one axis: physical units for formatted axis values.
    */
    std::string getUnit(int axis) const { return getC(detail::formatAxisAttr("Unit", axis)); }

    /**
    Find the point of intersection between two geodesic curves.

    For example, in a basic Frame, it will find the point of
    intersection between two straight lines. But for a SkyFrame it
    will find an intersection of two great circles.

    @warning This method can only be used with 2-dimensional Frames.

    @param[in] a1  Coordinates of the first point on the first geodesic curve.
    @param[in] a2  Coordinates of the second point on the first geodesic curve.
    @param[in] b1  Coordinates of the first point on the second geodesic curve.
    @param[in] b2  Coordinates of the second point on the second geodesic curve.

    @return the point of intersection between the two geodesic curves.

    @throws std::runtime_error if the frame is not 2-dimensional

    ### Notes

    - For a @ref SkyFrame each curve will be a great circle, and in general
    each pair of curves will intersect at two diametrically opposite
    points on the sky. The returned position is the one which is
    closest to point `a1`.
    - This method will return `nan` coordinate values
    if any of the input coordinates is invalid, or if the two
    points defining either geodesic are coincident, or if the two
    curves do not intersect.
    - The geodesic curve used by this method is the path of
    shortest distance between two points, as defined by @ref distance
    */
    std::vector<double> intersect(std::vector<double> const &a1, std::vector<double> const &a2,
                                  std::vector<double> const &b1, std::vector<double> const &b2) const;

    /**
    Look for corresponding axes between this frame and another.

    @param[in] other  The other frame

    @return The indices of the axes (within this Frame) that correspond to each axis within `other`.
        Axis indices start at 1. A value of zero will be stored in the returned array for each axis
        in `other` that has no corresponding axis in this frame.
        The number of elements in the array will be the number of axes in `other`.

    ### Notes

    - Corresponding axes are identified by the fact that a `Mapping` can be found between them
        using `findFrame` or `convert`. Thus, "corresponding axes" are not necessarily identical.
        For instance, `SkyFrame` axes will match even if they describe different celestial coordinate systems.
    */
    std::vector<int> matchAxes(Frame const &other) const {
        std::vector<int> ret(other.getNIn());
        astMatchAxes(getRawPtr(), other.getRawPtr(), ret.data());
        assertOK();
        return ret;
    }

    /**
    Combine this frame with another to form a compound frame (CmpFrame),
    with the axes of this frame followed by the axes of the `next` frame.

    A compound frame allows two component Frames
    (of any class) to be merged together to form a more complex
    Frame. The axes of the two component Frames then appear together
    in the resulting CmpFrame (those of this Frame, followed by those of `next`).

    Since a CmpFrame is itself a Frame, it can be used as a
    component in forming further CmpFrames. Frames of arbitrary
    complexity may be built from simple individual Frames in this
    way.

    Also since a Frame is a Mapping, a CmpFrame can also be used as a
    Mapping. Normally, a CmpFrame is simply equivalent to a UnitMap,
    but if either of the component Frames within a CmpFrame is a Region
    (a sub-class of Frame), then the CmpFrame will use the Region as a
    Mapping when transforming values for axes described by the Region.
    Thus input axis values corresponding to positions which are outside the
    Region will result in bad output axis values.

    The name comes the way vectors are sometimes shown for matrix multiplication:
    vertically, with the first axis at the bottom and the last axis at the top.

    @param[in]  next  The next frame in the compound frame (the final next.getNAxes() axes)
    @return a new CmpFrame

    @warning The contained frames are shallow copies (just like AST);
    if you want deep copies then make them manually.
    */
    CmpFrame under(Frame const &next) const;

    /**
    Normalise a set of Frame coordinate values which might be unsuitable for display
    (e.g. may lie outside the expected range) into a set of acceptable values suitable for display.

    @param[in] value  A point in the space which the Frame describes.

    @return Normalized version of `value`. If any axes of `value` lie outside the expected range
        for the Frame, the corresponding returned value is changed to a acceptable (normalised) value.
        Otherwise, the axes of `value` are returned unchanged.

    ### Notes
    - For some classes of Frame, whose coordinate values are not constrained, this function will
        always return the input unchanged. However, for Frames whose axes represent cyclic quantities
        (such as angles or positions on the sky), the output coordinates will typically be wrapped into
        an appropriate standard range, such as [0, 2 pi].
    - The NormMap class is a Mapping which can be used to normalise a set of points using the `norm` function
        of a specified Frame.
    - It is intended to be possible to put any set of coordinates into a form suitable for display
        by using this function to normalise them, followed by appropriate formatting (using astFormat).
    */
    PointD norm(PointD value) const {
        astNorm(getRawPtr(), value.data());
        assertOK();
        detail::astBadToNan(value);
        return value;
    }

    /**
    Find the point which is offset a specified distance along the geodesic curve between two other points.

    For example, in a basic Frame, this offset will be along the straight line joining two points.
    For a more specialised Frame describing a sky coordinate system, however, it would be along
    the great circle passing through two sky positions.

    @param[in] point1  The point marking the start of the geodesic curve.
    @param[in] point2  The point marking the end of the geodesic curve.
    @param[in] offset  The required offset from the first point along the geodesic curve.
        If this is positive, it will be towards the second point. If it is negative, it will be
        in the opposite direction. This offset need not imply a position lying between
        the two points given, as the curve will be extrapolated if necessary.

    @return the offset point

    ### Notes:
    - The geodesic curve used by this function is the path of shortest distance between two points,
        as defined by the `distance` function.

    @throws std::invalid_argument if:
    - point1 or point2 is the wrong length
    */
    PointD offset(PointD point1, PointD point2, double offset) const {
        assertPointLength(point1, "point1");
        assertPointLength(point2, "point2");
        PointD ret(getNIn());
        astOffset(getRawPtr(), point1.data(), point2.data(), offset, ret.data());
        assertOK();
        detail::astBadToNan(ret);
        return ret;
    }

    /**
    Find the point which is offset a specified distance along the geodesic curve at a given angle
    from a specified starting point. This can only be used with 2-dimensional Frames.

    For example, in a basic Frame, this offset will be along the straight line joining two points.
    For a more specialised Frame describing a sky coordinate system, however, it would be along
    the great circle passing through two sky positions.

    @param[in] point1  The point marking the start of the geodesic curve.
    @param[in] angle  The angle (in radians) from the positive direction of the second axis,
        to the direction of the required position, as seen from the starting position.
        Positive rotation is in the sense of rotation from the positive direction of axis 2
        to the positive direction of axis 1.
    @param[in] offset   The required offset from the first point along the geodesic curve.
        If this is positive, it will be in the direction of the given angle.  If it is negative, it
        will be in the opposite direction.

    @return a DirectionPoint containing:
    - The direction of the geodesic curve at the end point. That is, the angle (in radians)
        between the positive direction of the second axis and the continuation of the geodesic
        curve at the requested end point. Positive rotation is in the sense of rotation from
        the positive direction of axis 2 to the positive direction of axis 1.
    - The offset point.

    @throws std::invalid_argument if
    - frame does not have naxes = 2
    - point1 or point2 are the wrong length, or point1 has any `AST__BAD` values

    ### Notes
    - The geodesic curve used by this function is the path of shortest distance between two points,
        as defined by the astDistance function
    */
    DirectionPoint offset2(PointD const &point1, double angle, double offset) const {
        detail::assertEqual(getNIn(), "naxes", 2, " cannot call offset2");
        assertPointLength(point1, "point1");
        PointD point2(getNIn());
        double offsetAngle = astOffset2(getRawPtr(), point1.data(), angle, offset, point2.data());
        assertOK();
        detail::astBadToNan(point2);
        return DirectionPoint(detail::safeDouble(offsetAngle), point2);
    }

    /**
    Permute the order in which a Frame's axes occur

    @param[in] perm  A list of axes in their new order, using the current axis numbering.
        Axis numbers start at 1 for the first axis. Only genuine permutations of the axis order
        are permitted, so each axis must be referenced exactly once.

    When used on a @ref FrameSet, the axes of the current frame are permuted and all connecting
    mappings are updated accordingly, so that current behavior is preserved (except for the new
    axis order for output data).
    */
    void permAxes(std::vector<int> perm) {
        detail::assertEqual(perm.size(), "perm.size()", static_cast<std::size_t>(getNAxes()), "naxes");
        astPermAxes(getRawPtr(), perm.data());
        assertOK();
    }

    /**
    Create a new Frame whose axes are copied from an existing Frame along with other Frame attributes,
    such as its Title.

    Any number (zero or more) of the original Frame's axes may be copied, in any order,
    and additional axes with default attributes may also be included in the new Frame.

    @param[in] axes   the axes to be copied.  These should
                    be given in the order required in the new Frame, using the axis numbering in the
                    original Frame (which starts at 1 for the first axis).  Axes may be selected in
                    any order, but each may only be used once.  If additional (default) axes are also
                    to be included, the corresponding elements of this array should be set to zero.
    @return a FrameMapping containing:
    - The new frame.
    - A mapping (usually a PermMap, but may be a UnitMap) that describes the axis permutation
        that has taken place between the original and new Frames. The Mapping's forward
        transformation will convert coordinates from the original Frame into the new one,
        and vice versa.
    */
    FrameMapping pickAxes(std::vector<int> const &axes) const;

    /**
    Resolve a vector into two orthogonal components

    The vector from point 1 to point 2 is used as the basis vector.
    The vector from point 1 to point 3 is resolved into components
    parallel and perpendicular to this basis vector. The lengths of the
    two components are returned, together with the position of closest
    aproach of the basis vector to point 3.

    @param[in] point1  The start of the basis vector, and of the vector to be resolved.
    @param[in] point2  The end of the basis vector.
    @param[in] point3  The end of the vector to be resolved.

    @return a ResolvedPoint containing:
    - point  The point of closest approach of the basis vector to point 3.
    - d1  The distance from point 1 to the returned point
            (that is, the length of the component parallel
            to the basis vector). Positive values are in the same sense as
            movement from point 1 to point 2.
    - d2  The distance from the returned point to point 3
            (that is, the length of the component perpendicular to the basis vector).
            The value is always positive.

    ### Notes

    - Each vector used in this method is the path of
        shortest distance between two points, as defined by @ref distance
    - This method will return `nan` coordinate values
        if any of the input coordinates are invalid, or if the required
        output values are undefined.
    */
    ResolvedPoint resolve(std::vector<double> const &point1, std::vector<double> const &point2,
                          std::vector<double> const &point3) const;

    /**
    Set @ref Frame_AlignSystem "AlignSystem": the coordinate system used
    by @ref convert and @ref findFrame to align Frames
    */
    void setAlignSystem(std::string const &system) { setC("AlignSystem", system); }

    /**
    Set @ref Frame_Bottom "Bottom": the lowest axis value to display
    */
    void setBottom(int axis, double bottom) { setD(detail::formatAxisAttr("Bottom", axis), bottom); }

    /**
    Set @ref Frame_Digits "Digits" for all axes: number of digits of precision.
    */
    void setDigits(int digits) { setI("Digits", digits); }

    /**
    Set @ref Frame_Digits "Digits" for one axis: number of digits of precision.
    */
    void setDigits(int axis, int digits) { setD(detail::formatAxisAttr("Digits", axis), digits); }

    /**
    Set @ref Frame_Direction "Direction" for one axis: display axis in conventional direction?
    */
    void setDirection(bool direction, int axis) {
        setB(detail::formatAxisAttr("Direction", axis), direction);
    }

    /**
    Set @ref Frame_Domain "Domain": coordinate system domain
    */
    virtual void setDomain(std::string const &domain) { setC("Domain", domain); }

    /**
    Set @ref Frame_Dut1 "Dut1": difference between the UT1 and UTC timescale (sec)
    */
    void setDut1(double dut1) { setD("Dut1", dut1); }

    /**
    Set @ref Frame_Epoch "Epoch": Epoch of observation as a double (years)
    */
    void setEpoch(double epoch) { setD("Epoch", epoch); }

    /**
    Set @ref Frame_Epoch "Epoch": Epoch of observation as a string
    */
    void setEpoch(std::string const &epoch) { setC("Epoch", epoch); }

    /**
    Set @ref Frame_Format "Format" for one axis: format specification for axis values.
    */
    void setFormat(int axis, std::string const &format) {
        setC(detail::formatAxisAttr("Format", axis), format);
    }

    /**
    Set @ref Frame_Label "Label(axis)" for one axis: axis label.
    */
    void setLabel(int axis, std::string const &label) { setC(detail::formatAxisAttr("Label", axis), label); }

    /**
    Set @ref Frame_MatchEnd "MatchEnd": match trailing axes?
    */
    void setMatchEnd(bool match) { setB("MatchEnd", match); }

    /**
    Get @ref Frame_MaxAxes "MaxAxes": the maximum number of axes
    a frame found by @ref findFrame may have.
    */
    void setMaxAxes(int maxAxes) { setI("MaxAxes", maxAxes); }

    /**
    Get @ref Frame_MinAxes "MinAxes": the minimum number of axes
    a frame found by @ref findFrame may have.
    */
    void setMinAxes(int minAxes) { setI("MinAxes", minAxes); }

    /**
    Set @ref Frame_ObsAlt "ObsAlt": Geodetic altitude of observer (m).
    */
    void setObsAlt(double alt) { setD("ObsAlt", alt); }

    /**
    Set @ref Frame_ObsLat "ObsLat": frame title.
    */
    void setObsLat(std::string const &lat) { setC("ObsLat", lat); }

    /**
    Set @ref Frame_ObsLon "ObsLon": Geodetic longitude of observer.
    */
    void setObsLon(std::string const &lon) { setC("ObsLon", lon); }

    /**
    Set @ref Frame_ActiveUnit "ActiveUnit": pay attention to units when one @ref Frame
    is used to match another?
    */
    void setActiveUnit(bool enable) {
        astSetActiveUnit(getRawPtr(), enable);
        assertOK();
    }

    /**
    Set @ref Frame_Permute "Permute": allow axis permutation when used as a template?
    */
    void setPermute(bool permute) { setB("Permute", permute); }

    /**
    Set @ref Frame_PreserveAxes "PreserveAxes": preserve axes?
    */
    void setPreserveAxes(bool preserve) { setB("PreserveAxes", preserve); }

    /**
    Set @ref Frame_Symbol "Symbol(axis)" for one axis: axis symbol.
    */
    void setSymbol(int axis, std::string const &symbol) {
        setC(detail::formatAxisAttr("Symbol", axis), symbol);
    }

    /**
    Set @ref Frame_System "System": coordinate system used to describe
    positions within the domain.
    */
    void setSystem(std::string const &system) { setC("System", system); }

    /**
    Set @ref Frame_Title "Title": frame title.
    */
    void setTitle(std::string const &title) { setC("Title", title); }

    /**
    Set @ref Frame_Top "Top" for one axis: the highest axis value to display
    */
    void setTop(int axis, double top) { setD(detail::formatAxisAttr("Top", axis), top); }

    /**
    Set @ref Frame_Unit "Unit(axis)" for one axis: physical units for formatted axis values.
    */
    void setUnit(int axis, std::string const &unit) { setC(detail::formatAxisAttr("Unit", axis), unit); }

    /**
    Read a formatted coordinate value (given as a character string) for a Frame axis and return
    the number of characters read and the value.

    The principle use of this function is in decoding user-supplied input
    which contains formatted coordinate values. Free-format input is supported as far as possible.
    If input is ambiguous, it is interpreted with reference to the Frame's attributes
    (in particular, the Format string associated with the Frame's axis).

    This function is, in essence, the inverse of astFormat.

    Applicability:

    Frame
       This function applies to all Frames. See the "Frame Input
       Format" section below for details of the input formats
       accepted by a basic Frame.

    SkyFrame
       The SkyFrame class re-defines the input format to be suitable
       for representing angles and times, with the resulting
       coordinate value returned in radians.  See the "SkyFrame
       Input Format" section below for details of the formats
       accepted.

    @ref FrameSet
       The input formats accepted by a @ref FrameSet are determined by
       its current Frame (as specified by the Current attribute).

    Frame Input Format:

    The input format accepted for a basic Frame axis is as follows:
    - An optional sign, followed by:
    - A sequence of one or more digits possibly containing a decimal point, followed by:
    - An optional exponent field.
    - The exponent field, if present, consists of "E" or "e" followed by a possibly signed integer.

    Examples of acceptable Frame input formats include:
    - `99`
    - `1.25`
    - `-1.6`
    - `1E8`
    - `-.99e-17`
    - `<bad>`

    SkyFrame Input Format:

    The input format accepted for a SkyFrame axis is as follows:
    - An optional sign, followed by between one and three fields
    representing either degrees, arc-minutes, arc-seconds or hours,
    minutes, seconds (e.g. "-12 42 03").
    - Each field should consist of a sequence of one or more digits,
    which may include leading zeros. At most one field may contain a
    decimal point, in which case it is taken to be the final field
    (e.g. decimal degrees might be given as "124.707", while degrees
    and decimal arc-minutes might be given as "-13 33.8").
    - The first field given may take any value, allowing angles and
    times outside the conventional ranges to be
    represented. However, subsequent fields must have values of less
    than 60 (e.g. "720 45 31" is valid, whereas "11 45 61" is not).
    - Fields may be separated by white space or by ":" (colon), but
    the choice of separator must be used consistently throughout the
    value. Additional white space may be present around fields and
    separators (e.g. "- 2: 04 : 7.1").
    - The following field identification characters may be used as
    separators to replace either of those above (or may be appended
    to the final field), in order to identify the field to which
    they are appended: "d"---degrees; "h"---hours; "m"---minutes of
    arc or time; "s"---seconds of arc or time; "'" (single
    quote)---minutes of arc; """ (double quote)---seconds of arc.
    Either lower or upper case may be used.  Fields must be given in
    order of decreasing significance (e.g. "-11D 3' 14.4"" or
    "22h14m11.2s").
    - The presence of any of the field identification characters
    "d", "'" (single quote) or """ (double quote) indicates that the
    value is to be interpreted as an angle. Conversely, the presence
    of "h" indicates that it is to be interpreted as a time (with 24
    hours corresponding to 360 degrees). Incompatible angle/time
    identification characters may not be mixed (e.g. "10h14'3"" is
    not valid).  The remaining field identification characters and
    separators do not specify a preference for an angle or a time
    and may be used with either.
    - If no preference for an angle or a time is expressed anywhere
    within the value, it is interpreted as an angle if the Format
    attribute string associated with the SkyFrame axis generates an
    angle and as a time otherwise. This ensures that values produced
    by astFormat are correctly interpreted by Frame.unformat.
    - If no preference for an angle or a time is expressed anywhere
    within the value, it is interpreted as an angle if the Format
    attribute string associated with the SkyFrame axis generates an
    angle and as a time otherwise. This ensures that values produced
    by Frame.format are correctly interpreted by Frame.unformat.
    - Fields may be omitted, in which case they default to zero. The
    remaining fields may be identified by using appropriate field
    identification characters (see above) and/or by adding extra
    colon separators (e.g. "-05m13s" is equivalent to "-:05:13"). If
    a field is not identified explicitly, it is assumed that
    adjacent fields have been given, after taking account of any
    extra separator characters (e.g. "14:25.4s" specifies minutes
    and seconds, while "14::25.4s" specifies degrees and seconds).
    - If fields are omitted in such a way that the remaining ones
    cannot be identified uniquely (e.g. "01:02"), then the first
    field (either given explicitly or implied by an extra leading
    colon separator) is taken to be the most significant field that
    astFormat would produce when formatting a value (using the
    Format attribute associated with the SkyFrame axis).  By
    default, this means that the first field will normally be
    interpreted as degrees or hours. However, if this does not
    result in consistent field identification, then the last field
    (either given explicitly or implied by an extra trailing colon
    separator) is taken to to be the least significant field that
    astFormat would produce.
    - If fields are omitted in such a way that the remaining ones
    cannot be identified uniquely (e.g. "01:02"), then the first
    field (either given explicitly or implied by an extra leading
    colon separator) is taken to be the most significant field that
    Frame.format would produce when formatting a value (using the
    Format attribute associated with the SkyFrame axis).  By
    default, this means that the first field will normally be
    interpreted as degrees or hours. However, if this does not
    result in consistent field identification, then the last field
    (either given explicitly or implied by an extra trailing colon
    separator) is taken to to be the least significant field that
    Frame.format would produce.

    This final convention is intended to ensure that values formatted
    by Frame.format which contain less than three fields will be
    correctly interpreted if read back using Frame.unformat, even if
    they do not contain field identification characters.

    Examples of acceptable SkyFrame input formats (with
    interpretation in parentheses) include:
    - `-14d 13m 22.2s (-14d 13' 22.2")`
    - `+ 12:34:56.7` (12d 34' 56.7" or 12h 34m 56.7s)
    - `001 : 02 : 03.4` (1d 02' 03.4" or 1h 02m 03.4s)
    - `22h 30` (22h 30m 00s)
    - `136::10"` (136d 00' 10" or 136h 00m 10s)
    - `-14M 27S` (-0d 14' 27" or -0h 14m 27s)
    - `-:14:` (-0d 14' 00" or -0h 14m 00s)
    - `-::4.1` (-0d 00' 04.1" or -0h 00m 04.1s)
    - `.9"` (0d 00' 00.9")
    - `d12m` (0d 12' 00")
    - `H 12:22.3s` (0h 12m 22.3s)
    - `<bad>` (`AST__BAD`)

    Where alternative interpretations are shown, the choice of angle or
    time depends on the associated Format(axis) attribute.

    @param[in] axis  The number of the Frame axis for which the coordinate value
        is to be read (axis numbering starts at zero for the first axis).
    @param[in] str  String containing the formatted coordinate value.
    @return an NReadValue containing the number of characters read and the value;
        if nread is 0 then the value is certainly invalid.

    ### Notes
    - Any white space at the beginning of the string will be
    skipped, as also will any trailing white space following the
    coordinate value read. The number of characters read will reflect this.
    - The number of characters will be 0 and the value undefined
    if the string supplied does not contain a suitably formatted value.
    - The string "<bad>" is recognised as a special case and will
    generate the value AST__BAD, without error. The test for this
    string is case-insensitive and permits embedded white space.
    */
    NReadValue unformat(int axis, std::string const &str) const {
        double value;
        int nread = astUnformat(getRawPtr(), axis, str.c_str(), &value);
        assertOK();
        return NReadValue(nread, detail::safeDouble(value));
    }

protected:
    /**
    Construct a Frame from a pointer to a raw AstFrame.

    This method is public so subclasses can call it.

    @throws std::invalid_argument if `rawPtr` is not an AstFrame

    TODO make protected and use friend class
    */
    explicit Frame(AstFrame *rawPtr) : Mapping(reinterpret_cast<AstMapping *>(rawPtr)) {
        if (!astIsAFrame(getRawPtr())) {
            std::ostringstream os;
            os << "This is a " << getClassName() << ", which is not a Frame";
            throw std::invalid_argument(os.str());
        }
    }

    virtual std::shared_ptr<Object> copyPolymorphic() const override { return copyImpl<Frame, AstFrame>(); }

private:
    /**
    Assert that a point has the correct length

    @param[in] p  The point to check
    @param[in] name  Name of point to use in an error message if the point has the wrong length
    */
    template <typename T>
    void assertPointLength(T const &p, char const *name) const {
        if (static_cast<int>(p.size()) != getNIn()) {
            std::ostringstream os;
            os << "point " << name << " has " << p.size() << " axes, but " << getNIn() << " required";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
