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
#ifndef ASTSHIM_NORMMAP_H
#define ASTSHIM_NORMMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A @ref Mapping which normalises coordinate values using the @ref Frame.norm "norm" method
of the supplied @ref Frame. The number of inputs and outputs of a @ref NormMap
are both equal to the number of axes in the supplied @ref Frame.

The forward and inverse transformation of a @ref NormMap are both
defined but are identical (that is, they do not form a real inverse
pair in that the inverse transformation does not undo the
normalisation, instead it reapplies it). However, @ref Mapping.simplified
will replace neighbouring pairs of forward and inverse
@ref NormMap "NormMaps" by a single @ref UnitMap (so long as the @ref Frame "Frames" encapsulated by
the two @ref NormMap "NormMaps" are equal - i.e. have the same class and the same
attribute values). This means, for instance, that if a @ref CmpMap contains
a @ref NormMap, the @ref CmpMap will still cancel with its own inverse.

### Attributes

@ref NormMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class NormMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a NormMap

    @param[in] frame  Frame which is to be used to normalise the supplied axis values.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit NormMap(Frame const &frame, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      astNormMap(const_cast<AstObject *>(frame.getRawPtr()), "%s", options.c_str()))) {
                          assertOK();
                      }

    virtual ~NormMap() {}

    /// Copy constructor: make a deep copy
    NormMap(NormMap const &) = default;
    NormMap(NormMap &&) = default;
    NormMap &operator=(NormMap const &) = delete;
    NormMap &operator=(NormMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<NormMap> copy() const { return std::static_pointer_cast<NormMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<NormMap, AstNormMap>();
    }

    /// Construct a NormMap from a raw AST pointer
    explicit NormMap(AstNormMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsANormMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a NormMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
