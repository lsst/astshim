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
#ifndef ASTSHIM_PARALLELMAP_H
#define ASTSHIM_PARALLELMAP_H

#include <memory>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/CmpMap.h"

namespace ast {

/**
A parallel @ref CmpMap "compound mapping" where the first @ref Mapping is used
to transform the lower numbered coordinates of each point
and the second @ref Mapping is used to transform the remaining coordinates.

Since a @ref ParallelMap is itself a @ref Mapping, it can be used as a
component in forming further @ref ParallelMap "ParallelMaps".
@ref Mapping "Mappings" of arbitrary complexity may be built from simple
individual @ref Mapping "Mappings" in this way.

@warning ParallelMap is a convenience wrapper around CmpMap. Specialized code hides some
of this, so getClassName() will return "ParallelMap" and an ParallelMap persisted using a Channel
or pickle will be returned as a "ParallelMap" in Python. However, it will be visible in
other ways, such as the output from show().

### Attributes

@ref ParallelMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class ParallelMap : public CmpMap {
    friend class Object;

public:
    /**
    Construct a ParallelMap

    It may be clearer to constuct a @ref ParallelMap using @ref Mapping.under.

    @param[in] map1  The first mapping, which transforms the lower numbered coordinates/
    @param[in] map2  The second mapping.
    @param[in] options  Comma-separated list of attribute assignments.

    @warning @ref ParallelMap contains shallow copies of the provided mappings (just like AST).
    If you want a deep copy then copy the mapping before adding it.
    */
    explicit ParallelMap(Mapping const &map1, Mapping const &map2, std::string const &options = "")
            : CmpMap(map1, map2, false, options) {}

    virtual ~ParallelMap() {}

    /// Copy constructor: make a deep copy
    ParallelMap(ParallelMap const &) = default;
    ParallelMap(ParallelMap &&) = default;
    ParallelMap &operator=(ParallelMap const &) = delete;
    ParallelMap &operator=(ParallelMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<ParallelMap> copy() const {
        return std::static_pointer_cast<ParallelMap>(copyPolymorphic());
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<ParallelMap, AstCmpMap>();
    }

    /// Construct a ParallelMap from a raw AST pointer
    /// @todo add a test that the CmpMap is parallel
    explicit ParallelMap(AstCmpMap *rawptr) : CmpMap(rawptr) {
        if (getSeries()) {
            throw std::runtime_error("Compound mapping is in series");
        }
    }
};

}  // namespace ast

#endif
