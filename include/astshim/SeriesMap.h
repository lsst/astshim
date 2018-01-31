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
#ifndef ASTSHIM_SERIESMAP_H
#define ASTSHIM_SERIESMAP_H

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Mapping.h"
#include "astshim/CmpMap.h"

namespace ast {

/**
A series @ref CmpMap "compound mapping" where the first @ref Mapping is used
to transform the coordinates of each point and the second @ref Mapping
is then applied to the result.

Since a SeriesMap is itself a Mapping, it can be used as a
component in forming further @ref SeriesMap "SeriesMaps". @ref Mapping "Mappings" of arbitrary
complexity may be built from simple individual @ref Mapping "Mappings" in this way.

@warning SeriesMap is a convenience wrapper around CmpMap. Specialized code hides some
of this, so getClassName() will return "SeriesMap" and an SeriesMap persisted using a Channel
or pickle will be returned as a "SeriesMap" in Python. However, it will be visible in
other ways, such as the output from show().

### Attributes

@ref SeriesMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class SeriesMap : public CmpMap {
    friend class Object;

public:
    /**
    Construct a SeriesMap.

    It may be clearer to construct a @ref SeriesMap using @ref Mapping.then.

    @param[in] map1  The first mapping, which transforms input points.
    @param[in] map2  The second mapping, which transforms the output of the first mapping.
    @param[in] options  Comma-separated list of attribute assignments.

    @warning @ref SeriesMap contains shallow copies of the provided mappings (just like AST).
    If you want deep copies then provide deep copies to this constructor.
    */
    explicit SeriesMap(Mapping const &map1, Mapping const &map2, std::string const &options = "")
            : CmpMap(map1, map2, true, options) {}

    virtual ~SeriesMap() {}

    /// Copy constructor: make a deep copy
    SeriesMap(SeriesMap const &) = default;
    SeriesMap(SeriesMap &&) = default;
    SeriesMap &operator=(SeriesMap const &) = delete;
    SeriesMap &operator=(SeriesMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SeriesMap> copy() const {
        return std::static_pointer_cast<SeriesMap>(copyPolymorphic());
    }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SeriesMap, AstCmpMap>();
    }

    /// Construct a SeriesMap from a raw AST pointer
    explicit SeriesMap(AstCmpMap *rawptr) : CmpMap(rawptr) {
        if (!getSeries()) {
            throw std::runtime_error("Compound mapping is in parallel");
        }
    }
};

}  // namespace ast

#endif
