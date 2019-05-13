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
#ifndef ASTSHIM_FRAMESET_H
#define ASTSHIM_FRAMESET_H

#include <memory>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Frame.h"

namespace ast {

/**
A @ref FrameSet consists of a set of one or more @ref Frame "Frames" (which
describe coordinate systems), connected together by @ref Mapping "Mappings"
(which describe how the coordinate systems are inter-related).

A @ref FrameSet makes it possible to obtain a @ref Mapping between any pair
of these @ref Frame "Frames" (i.e. to convert between any of the coordinate
systems which it describes).  The individual @ref Frame "Frames" are
identified within the @ref FrameSet by an integer index, with @ref Frame "Frames"
being numbered consecutively from one as they are added to the
@ref FrameSet.

Every @ref FrameSet has a "base" @ref Frame and a "current" @ref Frame (which
are allowed to be the same). Any of the @ref Frame "Frames" may be nominated
to hold these positions, and the choice is determined by the
values of the @ref FrameSet's Base and Current attributes, which hold
the indices of the relevant @ref Frame "Frames".  By default, the first @ref Frame
added to a @ref FrameSet is its base @ref Frame, and the last one added is
its current @ref Frame.

The base @ref Frame describes the "native" coordinate system of
whatever the @ref FrameSet is used to calibrate (e.g. the pixel
coordinates of an image) and the current @ref Frame describes the
"apparent" coordinate system in which it should be viewed
(e.g. displayed, etc.). Any further @ref Frame "Frames" represent a library
of alternative coordinate systems, which may be selected by
making them current.

When a @ref FrameSet is used in a context that requires a @ref Frame,
(e.g. obtaining its Title value, or number of axes), the current
@ref Frame is used. A @ref FrameSet may therefore be used in place of its
current @ref Frame in most situations.

When a @ref FrameSet is used in a context that requires a @ref Mapping,
the @ref Mapping used is the one between its base @ref Frame and its
current @ref Frame. Thus, a @ref FrameSet may be used to convert "native"
coordinates into "apparent" ones, and vice versa. Like any
@ref Mapping, a @ref FrameSet may also be inverted (see @ref Mapping.inverted),
which has the effect of returning a copy with base and current @ref Frame "Frames"
swapped, hence of reversing the @ref Mapping between them.

Regions may be added into a @ref FrameSet (since a Region is a type of
@ref Frame), either explicitly or as components within @ref CmpFrame "CmpFrames". In this
case the @ref Mapping between a pair of @ref Frame "Frames" within a @ref FrameSet will
include the effects of the clipping produced by any Regions included
in the path between the @ref Frame "Frames".

### Attributes

In addition to those attributes common to @ref Frame @ref Mapping and @ref Object,
@ref FrameSet also has the following attributes:

- @ref FrameSet_AllVariants "AllVariants": a list of all variant mappings stored with the current @ref Frame
- @ref FrameSet_Base "Base": index of base @ref Frame, starting from 1
- @ref FrameSet_Current "Current": index of current @ref Frame, starting from 1
- @ref FrameSet_NFrame "NFrame": Number of Frames in a FrameSet
- @ref FrameSet_Variant "Variant": name of variant mapping in use by current Frame

Every FrameSet also inherits any further attributes that belong
to its current @ref Frame, regardless of that @ref Frame's class. (For
example, the `Equinox` attribute, defined by @ref SkyFrame, is
inherited by any FrameSet which has a @ref SkyFrame as its current
@ref Frame.) The set of attributes belonging to a FrameSet may therefore
change when a new current @ref Frame is selected.
*/
class FrameSet : public Frame {
    friend class Frame;  // so Frame can call the protected raw ptr constructor
    friend class Object;

public:
    static int constexpr BASE = AST__BASE;        ///< index of base frame
    static int constexpr CURRENT = AST__CURRENT;  ///< index of current frame
    static int constexpr NOFRAME = AST__NOFRAME;  ///< an invalid frame index
    /**
    Construct a FrameSet from a Frame

    The frame is deep copied.

    @param[in] frame  the first @ref Frame to be inserted into the @ref FrameSet.
                    This initially becomes both the base and the current Frame.
                    Further Frames may be added using @ref addFrame
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit FrameSet(Frame const &frame, std::string const &options = "")
            : FrameSet(astFrameSet(frame.copy()->getRawPtr(), "%s", options.c_str())) {
        assertOK();
    }

    /**
    Construct a FrameSet from two frames and a mapping that connects them

    Both frames and the mapping are deep copied.

    @param[in] baseFrame  base @ref Frame.
    @param[in] mapping  mapping connecting baseFrame to currentFrame.
    @param[in] currentFrame  current @ref Frame.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit FrameSet(Frame const &baseFrame, Mapping const &mapping, Frame const &currentFrame,
                      std::string const &options = "")
            : FrameSet(baseFrame, options) {
        _basicAddFrame(1, mapping, currentFrame);
    }

    virtual ~FrameSet() {}

    /// Copy constructor: make a deep copy
    FrameSet(FrameSet const &) = default;
    FrameSet(FrameSet &&) = default;
    FrameSet &operator=(FrameSet const &) = delete;
    FrameSet &operator=(FrameSet &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<FrameSet> copy() const { return std::static_pointer_cast<FrameSet>(copyPolymorphic()); }

    /**
    Append the axes from a specified @ref Frame to every existing @ref Frame in this FrameSet.

    In detail, each @ref Frame in this FrameSet is replaced by a CmpFrame containing the original @ref Frame
    and the @ref Frame specified by parameter `frame`. In addition, each @ref Mapping in the FrameSet
    is replaced by a CmpMap containing the original @ref Mapping and a UnitMap in parallel.
    The NIn and NOut attributes of the UnitMap are set equal to the number of axes
    in the supplied @ref Frame.  Each new CmpMap is simplified before being stored in the FrameSet.

    @param[in] frame  @ref Frame whose axes are to be appended to each @ref Frame in this FrameSet.
    */
    void addAxes(Frame const &frame) {
        astAddFrame(getRawPtr(), AST__ALLFRAMES, nullptr, frame.getRawPtr());
        assertOK();
    }

    /**
    Add a new Frame and an associated @ref Mapping to this FrameSet so as to define a new coordinate system,
    derived from one which already exists within this FrameSet.

    If `frame` is a Frame then it becomes the current frame and its index is the new number of frames.
    If `frame` is a FrameSet then its current frame becomes the new current frame and the indices
    of all its frames are increased by the number of frames originally in this FrameSet.
    In both cases the indices of the Frames already in this FrameSet are left unchanged.

    @param[in] iframe  The index of the Frame within the FrameSet which describes the coordinate system
        upon which the new one is to be based.  This value should lie in the range from
        1 to the number of frames already in this FrameSet (as given by @ref getNFrame).
        A value of FrameSet::BASE or FrameSet::CURRENT may be given to specify the base @ref Frame
        or the current @ref Frame respectively.
        A value of `AST__ALLFRAMES` is not permitted; call `addAllFrames` instead.
    @param[in] map  A @ref Mapping which describes how to convert coordinates from the old coordinate
        system (described by the @ref Frame with index `iframe` ) into coordinates in the
        new system.  The Mapping's forward transformation should perform this conversion,
        and its inverse transformation should convert in the opposite direction.
    @param[in] frame  A @ref Frame that describes the new coordinate system.  Any type of @ref Frame
        may be supplied (including Regions and FrameSets).
        This function may also be used to merge two FrameSets by supplying a pointer to
        a second FrameSet for this parameter (see the Notes section for details).

    ### Notes

    - Deep copies of the supplied `map` and `frame` are stored within the modified FrameSet.
        So any changes made to the FrameSet after calling this method will have no effect on the supplied
        @ref Mapping and @ref Frame objects.
    - This function sets the value of the `Current` attribute for the FrameSet so that the new @ref Frame
        subsequently becomes the current @ref Frame.
    - The number of input coordinate values accepted by the supplied `map` (its NIn attribute)
        must match the number of axes in the @ref Frame identified by the `iframe` parameter.
        Similarly, the number of output coordinate values generated by `map` (its NOut attribute)
        must match the number of axes in `frame`.
    - As a special case, if a pointer to a FrameSet is given for the `frame` parameter,
        this is treated as a request to merge `frame` into this FrameSet. This is done by appending
        all the new @ref Frame "Frames" in the `frame` FrameSet to this FrameSet,
        while preserving their order and retaining all the inter-relationships (i.e. @ref Mapping "Mappings")
        between them. The two sets of @ref Frame "Frames"
        are inter-related within the merged FrameSet by using the @ref Mapping supplied.
        This should convert between the @ref Frame identified by the `iframe` parameter (in this FrameSet)
        and the `current` @ref Frame of the `frame` FrameSet. This latter @ref Frame becomes
        the current @ref Frame in this FrameSet.
    */
    virtual void addFrame(int iframe, Mapping const &map, Frame const &frame) {
        _basicAddFrame(iframe, map, frame);
    }

    /**
    Store a new variant @ref Mapping with the @ref getCurrent "current" @ref Frame.

    The newly added variant becomes the current variant (attribute
    the @ref FrameSet_Variant "Variant" is set to `name`).

    See the @ref FrameSet_Variant "Variant" attribute for more details.
    See also @ref getVariant, @ref renameVariant and @ref mirrorVariants

    @param[in] map  A @ref Mapping which describes how to convert coordinates
                    from the @ref getCurrent "current" frame
                    to the new variant of the @ref getCurrent "current" Frame.
    @param[in] name  The name to associate with the new variant Mapping.

    @throws std::runtime_error if:
    - A variant with the supplied name already exists in the current Frame.
    - The current Frame is a mirror for the variant Mappings in another Frame.
        This is only the case if the astMirrorVariants function has been called
        to make the current Frame act as a mirror.
    */
    void addVariant(Mapping const &map, std::string const &name) {
        astAddVariant(getRawPtr(), map.getRawPtr(), name.c_str());
        assertOK();
    }

    /**
    Get @ref FrameSet_AllVariants "AllVariants": a list of all variant mappings
    stored with the current @ref Frame
    */
    std::string getAllVariants() const { return getC("AllVariants"); }

    /**
    Get @ref FrameSet_Base "Base": index of base @ref Frame
    */
    int getBase() const { return getI("Base"); }

    /**
    Get @ref FrameSet_Current "Current": index of current @ref Frame, starting from 1
    */
    int getCurrent() const { return getI("Current"); }

    /**
    Obtain a deep copy of the specified @ref Frame

    @param[in] iframe  The index of the required @ref Frame within this @ref FrameSet.
        This value should lie in the range 1 to the number of frames already in this @ref FrameSet
        (as given by @ref getNFrame). A value of FrameSet::Base or FrameSet::CURRENT
        may be given to specify the base @ref Frame or the current @ref Frame, respectively.
    @param[in] copy  If true return a deep copy, else a shallow copy.

    @warning: to permute axes of a frame in a FrameSet, such that
    the connecting mappings are updated: set the current frame
    to the frame in question and call `permAxes` directly on the FrameSet.
    Do *not* call `permAxes` on a shallow copy of the frame (retrieved
    by getFrame) as this will not affect the connected mappings.
    */
    std::shared_ptr<Frame> getFrame(int iframe, bool copy = true) const {
        auto *rawFrame = reinterpret_cast<AstObject *>(astGetFrame(getRawPtr(), iframe));
        assertOK(rawFrame);
        if (!rawFrame) {
            throw std::runtime_error("getFrame failed (returned a null frame)");
        }
        return Object::fromAstObject<Frame>(rawFrame, copy);
    }

    /**
    Obtain a @ref Mapping that converts between two @ref Frame "Frames" in a @ref FrameSet

    @param[in] from   The index of the first @ref Frame in the @ref FrameSet, the frame
        describing the coordinate system for the "input" end of the Mapping.
        This value should lie in the range 1 to the number of frames already in this @ref FrameSet
        (as given by @ref getNFrame).
    @param[in] to   The index of the second @ref Frame in the @ref FrameSet,
        the frame describing the coordinate system for the "output" end of the @ref Mapping.
        This value should lie in the range 1 to the number of frames already in this @ref FrameSet
        (as given by @ref getNFrame).

    @return A @ref Mapping whose forward transformation converts coordinates from the first
        frame to the second one, and whose inverse transformation converts coordinates
        in the opposite direction.

    ### Notes

    - The returned @ref Mapping will include the clipping effect of any Regions which occur on the path
        between the two supplied @ref Frame "Frames", including the specified end frames.
    - It should always be possible to generate the @ref Mapping requested, but this does not
        necessarily guarantee that it will be able to perform the required coordinate conversion.
        If necessary, call `hasForward` or `hasInverse` on the returned @ref Mapping
        to determine if the required transformation is available.
    */
    std::shared_ptr<Mapping> getMapping(int from = BASE, int to = CURRENT) const {
        AstObject *rawMap = reinterpret_cast<AstObject *>(astGetMapping(getRawPtr(), from, to));
        assertOK(rawMap);
        if (!rawMap) {
            throw std::runtime_error("getMapping failed (returned a null mapping)");
        }
        return Object::fromAstObject<Mapping>(rawMap, true);
    }

    /**
    Get FrameSet_NFrame "NFrame": number of @ref Frame "Frames" in the @ref FrameSet, starting from 1
    */
    int getNFrame() const { return getI("NFrame"); }

    /**
    @ref FrameSet_Variant "Variant": name of variant mapping in use by current Frame

    See also @ref addVariant, @ref mirrorVariants and @ref renameVariant
    */
    std::string getVariant() const { return getC("Variant"); }

    /**
    Indicates that all access to the @ref FrameSet_Variant "Variant"
    attribute of the current @ref Frame should should be forwarded
    to some other nominated @ref Frame in the @ref FrameSet.

    For instance, if a value is set subsequently for the
    Variant attribute of the current @ref Frame, the current @ref Frame will be left
    unchanged and the setting is instead applied to the nominated @ref Frame.
    Likewise, if the value of the Variant attribute is requested, the
    value returned is the value stored for the nominated @ref Frame rather
    than the current @ref Frame itself.

    This provides a mechanism for propagating the effects of variant
    Mappings around a @ref FrameSet. If a new @ref Frame is added to a @ref FrameSet
    by connecting it to an pre-existing @ref Frame that has two or more variant
    Mappings, then it may be appropriate to set the new @ref Frame so that it
    mirrors the variants Mappings of the pre-existing @ref Frame. If this is
    done, then it will be possible to select a specific variant Mapping
    using either the pre-existing @ref Frame or the new @ref Frame.

    See also @ref addVariant, @ref getVariant and @ref renameVariant.

    @param[in] iframe
       The index of the @ref Frame within the @ref FrameSet which is to be
       mirrored by the current @ref Frame. This value should lie in the range
       from 1 to the number of @ref Frame "Frames"
       in the @ref FrameSet (as given by @ref getNFrame).
       If `AST__NOFRAME` is supplied (or the current @ref Frame is specified),
       then any mirroring established by a previous call to this function is disabled.
       A value of FrameSet::BASE may be given to specify the base frame.

    ### Notes:
    - Mirrors can be chained. That is, if @ref Frame B is set to be a mirror
        of @ref Frame A, and @ref Frame C is set to be a mirror of @ref Frame B, then
        @ref Frame C will act as a mirror of @ref Frame A.
    - Variant Mappings cannot be added to the current @ref Frame if it is
        mirroring another @ref Frame. So calls to @ref addVariant
        will cause an error to be reported if the current @ref Frame is
        mirroring another @ref Frame.
    - Any variant Mappings explicitly added to the current @ref Frame using @ref addVariant
        will be ignored if the current @ref Frame is mirroring another @ref Frame.
    */
    void mirrorVariants(int iframe) {
        astMirrorVariants(getRawPtr(), iframe);
        assertOK();
    }

    /**
    Modify the relationship (i.e. @ref Mapping) between a specified @ref Frame in a @ref FrameSet
    and the other @ref Frame "Frames" in that @ref FrameSet.

    Typically, this might be required if the @ref FrameSet has been used
    to calibrate (say) an image, and that image is re-binned. The
    @ref Frame describing the image will then have undergone a coordinate
    transformation, and this should be communicated to the associated
    @ref FrameSet using this function.

    See also @ref addVariant, @ref getVariant and @ref mirrorVariants.


    @param[in] iframe  The index within the @ref FrameSet of the @ref Frame to be modified.
        This value should lie in the range
        1 to the number of frames already in this @ref FrameSet (as given by @ref getNFrame).
        A value of FrameSet::BASE or FrameSet::CURRENT may be given
        to specify the base @ref Frame or the current @ref Frame respectively.
    @param[in] map  A @ref Mapping whose forward transformation converts coordinate values from
        the original coordinate system described by the @ref Frame to the new one, and whose
        inverse transformation converts in the opposite direction.

    ### Notes

    - The relationship between the selected @ref Frame and any other @ref Frame
        within the @ref FrameSet will be modified by this function, but the relationship between
        all other @ref Frame "Frames" in the @ref FrameSet remains unchanged.
    - The number of input and output coordinate values of the @ref Mapping
        must be equal and must match the number of axes in the @ref Frame being modified.
    - If a simple change of axis order is required, then @ref permAxes may provide
        a more straightforward method of making the required changes to the @ref FrameSet.
    - This function cannot be used to change the number of @ref Frame axes.
        To achieve this, a new @ref Frame must be added to the @ref FrameSet (with @ref addFrame)
        and the original one removed if necessary (with @ref removeFrame).
    - Any variant @ref Mapping "Mappings" associated with the remapped @ref Frame
        (except for the current variant) will be lost as a consequence of calling this method
        (see attribute `Variant`).
    */
    void remapFrame(int iframe, Mapping &map) {
        astRemapFrame(getRawPtr(), iframe, map.copy()->getRawPtr());
        assertOK();
    };

    /**
    Remove a @ref Frame from a @ref FrameSet

    Other Frame indices in the FrameSet are re-numbered as follows: Frame indices greater than `iframe`
    are decremented by one; other Frame indeces retain the same index.

    @param[in] iframe  The index of the required @ref Frame within this @ref FrameSet.
        This value should lie in the range 1 to the number of @ref Frame "Frames"
        in this @ref FrameSet (as given by @ref getNFrame).
        A value of FrameSet::BASE or FrameSet::CURRENT may be given
        to specify the base @ref Frame or the current @ref Frame, respectively.

    ### Notes

    - Removing a @ref Frame from a @ref FrameSet does not affect the relationship between
        other @ref Frame "Frames" in this @ref FrameSet,
        even if they originally depended on the @ref Frame being removed.
    - The number of @ref Frame "Frames" in a @ref FrameSet cannot be reduced to zero.
    - If a @ref FrameSet's base or current Frame is removed, the `Base` or `Current` attribute (respectively)
        of the @ref FrameSet will have its value cleared, so that another @ref Frame will then assume its role
        by default.
    - If any other @ref Frame is removed, the base and current @ref Frame "Frames" will remain the same.
        To ensure this, the `Base` and/or `Current` attributes of the @ref FrameSet will be changed,
        if necessary, to reflect any change in the indices of these @ref Frame "Frames".

    @throws std::runtime_error if you attempt to remove the last frame
    */
    virtual void removeFrame(int iframe) {
        astRemoveFrame(getRawPtr(), iframe);
        assertOK();
    }

    /**
    Rename the current @ref FrameSet_Variant "Variant" of the current @ref Mapping

    The @ref FrameSet_Variant "Variant" attribute is updated to `name`.

    See the @ref FrameSet_Variant "Variant" attribute for more details.
    See also @ref addVariant, @ref getVariant and @ref mirrorVariants.

    @param[in] name  The new name of the current variant Mapping.

    @throws std::runtime_error if:
    - A variant with the supplied name already exists in the current Frame.
    - The current Frame is a mirror for the variant Mappings in another Frame.
        This is only the case if the astMirrorVariants function has been called
        to make the current Frame act as a mirror.
    */
    void renameVariant(std::string const &name) {
        astAddVariant(getRawPtr(), nullptr, name.c_str());
        assertOK();
    }

    /**
    Set @ref FrameSet_Base "Base": index of base @ref Frame
    */
    void setBase(int ind) { setI("Base", ind); }

    /**
    Set @ref FrameSet_Current "Current": index of current @ref Frame, starting from 1
    */
    void setCurrent(int ind) { setI("Current", ind); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<FrameSet, AstFrameSet>();
    }

    /**
    Construct a FrameSet from a raw AST pointer

    This method is public so @ref Frame can call it.

    @throws std::invalid_argument if `rawPtr` is not an AstFrameSet.
    */
    explicit FrameSet(AstFrameSet *rawPtr) : Frame(reinterpret_cast<AstFrame *>(rawPtr)) {
        if (!astIsAFrameSet(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a FrameSet";
            throw std::invalid_argument(os.str());
        }
    }

private:
    // non-virtual version of addFrame for use by constructors
    void _basicAddFrame(int iframe, Mapping const &map, Frame const &frame) {
        if (iframe == AST__ALLFRAMES) {
            throw std::runtime_error("iframe = AST__ALLFRAMES; call addAxes instead");
        }
        // astAddFrame makes deep copies of the map and frame, so no need to do anything extra
        astAddFrame(getRawPtr(), iframe, map.getRawPtr(), frame.getRawPtr());
        assertOK();
    }

};

}  // namespace ast

#endif
