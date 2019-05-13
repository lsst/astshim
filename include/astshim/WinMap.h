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
#ifndef ASTSHIM_WINMAP_H
#define ASTSHIM_WINMAP_H

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A WinMap is a linear Mapping which transforms a rectangular
window in one coordinate system into a similar window in another
coordinate system by scaling and shifting each axis (the window
edges being parallel to the coordinate axes).

### Attributes

@ref WinMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class WinMap : public Mapping {
    friend class Object;

public:
    /**
    Create a WinMap from the coordinates of two opposite corners (A and B)
    of the window in both the input and output coordinate systems.

    @param[in] ina   Coordinates of corner A of the window in the input coordinate system.
    @param[in] inb   Coordinates of corner B of the window in the input coordinate system.
    @param[in] outa   Coordinates of corner A of the window in the output coordinate system.
    @param[in] outb   Coordinates of corner B of the window in the output coordinate system.
    @param[in] options  Comma-separated list of attribute assignments.

    @throws std::invalid_argument if the lengths of the input vectors do not all match.
    */
    explicit WinMap(std::vector<double> const &ina, std::vector<double> const &inb,
                    std::vector<double> const &outa, std::vector<double> const &outb,
                    std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(_makeRawWinMap(ina, inb, outa, outb, options))) {}

    virtual ~WinMap() {}

    /// Copy constructor: make a deep copy
    WinMap(WinMap const &) = default;
    WinMap(WinMap &&) = default;
    WinMap &operator=(WinMap const &) = delete;
    WinMap &operator=(WinMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<WinMap> copy() const { return std::static_pointer_cast<WinMap>(copyPolymorphic()); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<WinMap, AstWinMap>();
    }

    /// Construct a WinMap from a raw AST pointer
    explicit WinMap(AstWinMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAWinMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a WinMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    AstWinMap *_makeRawWinMap(std::vector<double> const &ina, std::vector<double> const &inb,
                              std::vector<double> const &outa, std::vector<double> const &outb,
                              std::string const &options = "") {
        auto const ncoord = ina.size();
        if (inb.size() != ncoord) {
            std::ostringstream os;
            os << "inb.size() = " << inb.size() << " != " << ncoord << " = ina.size()";
            throw std::invalid_argument(os.str());
        }
        if (outa.size() != ncoord) {
            std::ostringstream os;
            os << "outa.size() = " << outa.size() << " != " << ncoord << " = ina.size()";
            throw std::invalid_argument(os.str());
        }
        if (outb.size() != ncoord) {
            std::ostringstream os;
            os << "outb.size() = " << outb.size() << " != " << ncoord << " = ina.size()";
            throw std::invalid_argument(os.str());
        }
        auto result = astWinMap(static_cast<int>(ncoord), ina.data(), inb.data(), outa.data(),
                                outb.data(), "%s", options.c_str());
        assertOK();
        return result;
    }
};

}  // namespace ast

#endif
