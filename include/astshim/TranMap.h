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
#ifndef ASTSHIM_TRANMAP_H
#define ASTSHIM_TRANMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
TranMap is a Mapping which combines the forward transformation of
a supplied Mapping with the inverse transformation of another
supplied Mapping, ignoring the un-used transformation in each
Mapping (indeed the un-used transformation need not exist).

When the forward transformation of the TranMap is referred to, the
transformation actually used is the forward transformation of the
first Mapping supplied when the TranMap was constructed. Likewise,
when the inverse transformation of the TranMap is referred to, the
transformation actually used is the inverse transformation of the
second Mapping supplied when the TranMap was constructed.

### Attributes

@ref TranMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class TranMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref TranMap

    @param[in] map1  The first component Mapping, which defines the forward transformation.
    @param[in] map2  The second component Mapping, which defines the inverse transformation.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit TranMap(Mapping const &map1, Mapping const &map2, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astTranMap(const_cast<AstObject *>(map1.getRawPtr()),
                                                                const_cast<AstObject *>(map2.getRawPtr()),
                                                                "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~TranMap() {}

    /// Copy constructor: make a deep copy
    TranMap(TranMap const &) = default;
    TranMap(TranMap &&) = default;
    TranMap &operator=(TranMap const &) = delete;
    TranMap &operator=(TranMap &&) = default;

    /**
    Return a shallow copy of one of the two component mappings.

    @param[in] i  Index: 0 for the forward mapping, 1 for the inverse.
    @throws std::invalid_argument if `i` is not 0 or 1.
    */
    std::shared_ptr<Mapping> operator[](int i) const { return decompose<Mapping>(i, false); };

    /// Return a deep copy of this object.
    std::shared_ptr<TranMap> copy() const { return std::static_pointer_cast<TranMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<TranMap, AstTranMap>();
    }

    /// Construct a TranMap from a raw AST pointer
    explicit TranMap(AstTranMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsATranMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a TranMap";
            throw std::invalid_argument(os.str());
        }
    }
};
}  // namespace ast

#endif
