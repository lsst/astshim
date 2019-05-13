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
#ifndef ASTSHIM_RATEMAP_H
#define ASTSHIM_RATEMAP_H

#include <algorithm>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
RateMap is a @ref Mapping which represents a single element of the
Jacobian matrix of another @ref Mapping. The @ref Mapping for which the
Jacobian is required is specified when the new RateMap is created,
and is referred to as the "encapsulated @ref Mapping" below.

The number of inputs to a RateMap is the same as the number of inputs
to its encapsulated @ref Mapping. The number of outputs from a RateMap
is always one. This one output equals the rate of change of a
specified output of the encapsulated @ref Mapping with respect to a
specified input of the encapsulated @ref Mapping (the input and output
to use are specified when the RateMap is created).

A RateMap which has not been inverted does not define an inverse
transformation. If a RateMap has been inverted then it will define
an inverse transformation but not a forward transformation.

### Attributes

@ref RateMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class RateMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref RateMap

    @param[in] map  The encapsulated mapping.
    @param[in] ax1  Index of the output from the encapsulated @ref Mapping for which the
                    rate of change is required. This corresponds to the delta
                    quantity forming the numerator of the required element of the
                    Jacobian matrix. The first axis has index 1.
    @param[in] ax2  Index of the input to the encapsulated @ref Mapping which is to be
                    varied. This corresponds to the delta quantity forming the
                    denominator of the required element of the Jacobian matrix.
                    The first axis has index 1.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit RateMap(Mapping const &map, int ax1, int ax2, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astRateMap(const_cast<AstObject *>(map.getRawPtr()), ax1,
                                                                ax2, "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~RateMap() {}

    /// Copy constructor: make a deep copy
    RateMap(RateMap const &) = default;
    RateMap(RateMap &&) = default;
    RateMap &operator=(RateMap const &) = delete;
    RateMap &operator=(RateMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<RateMap> copy() const { return std::static_pointer_cast<RateMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<RateMap, AstRateMap>();
    }

    /// Construct a RateMap from a raw AST pointer
    explicit RateMap(AstRateMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsARateMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a RateMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
