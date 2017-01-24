/* 
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#ifndef ASTSHIM_DETAIL_H
#define ASTSHIM_DETAIL_H

#include <stdexcept>

#include "astshim/base.h"

namespace ast {
namespace detail {

static const int FITSLEN=80;

/// A wrapper around astAnnul; intended as a custom deleter for std::unique_ptr
inline void annulAstObject(AstObject * object) {
    if (object != nullptr) {
        astAnnul(object);
    }
}

template<typename T1, typename T2>
inline void assertEqual(T1 val1, std::string const & descr1, T2 val2, std::string const & descr2) {
    if (val1 != val2) {
        std::ostringstream os;
        os << descr1 << " = " << val1 << " != " << descr2 << " = " << val2;
        throw std::invalid_argument(os.str());
    }
}

/**
Replace `AST__BAD` with a quiet NaN in a vector
*/
inline void astBadToNan(std::vector<double> & p) {
    for (auto & val: p) {
        if (val == AST__BAD) {
            val = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

/**
Replace `AST__BAD` with a quiet NaN in a vector
*/
inline void astBadToNan(ast::Array2D & arr) {
    for (auto i = arr.begin(); i != arr.end(); ++i) {
        for (auto j = i->begin(); j != i->end(); ++j) {
            if (*j == AST__BAD) {
                *j = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

/**
Format an axis-specific attribute by appending the axis index

@param[in] name  Attribute name
@param[in] axis  Axis index, starting at 1
@return "<name>(<axis>)"
*/
inline std::string formatAxisAttr(std::string const & name, int axis) {
    std::stringstream os;
    os << name << "(" << axis << ")";
    return os.str();
}

/**
Make a shallow copy of a raw AST pointer without checking

Intended for use in cast constructors, e.g.:
    explicit FrameSet(Object const & obj) : FrameSet(detail::shallowCopy<AstFrameSet>(obj.getRawPtr())) {}

There is probably a cleaner way to do this by using the shared_ptr contained
in each Object (thus avoiding the need to call astClone).
*/
template<typename AstT>
AstT * shallowCopy(AstObject * rawPtr) {
    return reinterpret_cast<AstT *>(astClone(rawPtr));
}

/**
Return a double value after checking status and replacing `AST__BAD` with `nan`
*/
inline double safeDouble(double val) {
    assertOK();
    return val != AST__BAD ? val : std::numeric_limits<double>::quiet_NaN();
}

}}  // namespace ast::detail

#endif