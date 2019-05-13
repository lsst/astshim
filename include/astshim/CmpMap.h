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
#ifndef ASTSHIM_CMPMAP_H
#define ASTSHIM_CMPMAP_H

#include <memory>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Mapping.h"

namespace ast {

/**
Abstract base class for @ref SeriesMap and @ref ParallelMap

@ref CmpMap is a compound @ref Mapping which allows two component
@ref Mapping "Mappings" (of any class) to be connected together to form a more
complex @ref Mapping. This connection may either be "in series"
(where the first @ref Mapping is used to transform the coordinates of
each point and the second mapping is then applied to the
result), or "in parallel" (where the first @ref Mapping transforms the
lower numbered coordinates for each point and the second @ref Mapping
simultaneously transforms the higher numbered coordinates).

Since a @ref CmpMap is itself a @ref Mapping, it can be used as a
component in forming further CmpMaps. @ref Mapping "Mappings" of arbitrary
complexity may be built from simple individual @ref Mapping "Mappings" in this
way.

### Attributes

@ref CmpMap has no attributes beyond those provided by @ref Mapping and @ref Object.

@warning CmpMap will sometimes appears as a SeriesMap or ParallelMap, as appropriate, including:
- getClassName() will return "SeriesMap" or "ParallelMap", as appropriate
- A CmpMap persisted using a Channel or pickle will be unpersisted as a SeriesMap or ParallelMap
*/
class CmpMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref CmpMap

    @param[in] map1  First mapping. When calling @ref Mapping.applyForward
        the first mapping is applied to the input points in a series mapping,
        and to the lower numbered coordinates in a parallel mapping.
    @param[in] map2  Second mapping.
    @param[in] series  Is this a series mapping?
    @param[in] options  Comma-separated list of attribute assignments.

    @warning @ref CmpMap contains shallow copies of the provided mappings (just like AST).
    If you deep copies then provide deep copies to this constructor.
    */
    explicit CmpMap(Mapping const &map1, Mapping const &map2, bool series, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astCmpMap(const_cast<AstObject *>(map1.getRawPtr()),
                                                               const_cast<AstObject *>(map2.getRawPtr()),
                                                               series, "%s", options.c_str()))) {
        assertOK();
    }

    virtual ~CmpMap() {}

    /// Copy constructor: make a deep copy
    CmpMap(CmpMap const &) = default;
    CmpMap(CmpMap &&) = default;
    CmpMap &operator=(CmpMap const &) = delete;
    CmpMap &operator=(CmpMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<CmpMap> copy() const { return std::static_pointer_cast<CmpMap>(copyPolymorphic()); }

    /**
    Return a shallow copy of one of the two component mappings.

    @param[in] i  Index: 0 for the first mapping, 1 for the second.
    @throws std::invalid_argument if `i` is not 0 or 1.
    */
    std::shared_ptr<Mapping> operator[](int i) const { return decompose<Mapping>(i, false); };

    /// Return True if the map is in series
    bool getSeries() { return detail::isSeries(reinterpret_cast<AstCmpMap *>(getRawPtr())); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<CmpMap, AstCmpMap>();
    }

    /// Construct a @ref CmpMap from a raw AST pointer
    /// (protected instead of private so that SeriesMap and ParallelMap can call it)
    explicit CmpMap(AstCmpMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsACmpMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a CmpMap";
            throw std::invalid_argument(os.str());
        }
    }
};

}  // namespace ast

#endif
