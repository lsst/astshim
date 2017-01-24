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
#ifndef ASTSHIM_PERMMAP_H
#define ASTSHIM_PERMMAP_H

#include <memory>
#include <algorithm>  // for std::max
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace {

/**
Throw std::invalid_argument if a permutation array calls for a constant that is not available

@param[in] numConst  Size of constant array
@param[in] perm  Permutation vector (inperm or outperm)
@param[in] name  Name of permutation vector, to use in reporting a problem
*/
void checkConstant(int numConst, std::vector<int> const & perm, std::string const & name) {
    int maxConst = 0;
    for (int const & innum : perm) {
        maxConst = std::max(maxConst, -innum);
    }
    if (maxConst > numConst) {
        std::ostringstream os;
        os << name << " specifies max constant number (min negative number) " << maxConst
            << ", but only " << numConst << " constants are available";
        throw std::invalid_argument(os.str());
    }
}

}  // anonymous namespace

namespace ast {

/**
A @ref Mapping which permutes the order of coordinates, and possibly also changes
the number of coordinates, between its input and output.

In addition to permuting the coordinate order, a PermMap may also assign constant values to coordinates.
This is useful when the number of coordinates is being increased as it allows fixed values to be assigned
to any new ones.

### Attributes

@ref PermMap has no attributes beyond those provided by @ref Mapping and @ref Object.
*/
class PermMap: public Mapping {
friend class Object;
public:
    /**
    Construct a PermMap

    Unlike AST's astPermMap, you must always provide non-empty inperm and outperm vectors.

    @param[in] inperm  A vector of `nin` elements; each element specifies the number of the
            output coordinate whose value is to be used (note that this array
            therefore defines the inverse coordinate transformation).
            Coordinates are numbered starting from 1.
    @param[in] outperm  A vector of `nout` elements; each element specifies the number of the
            input coordinate whose value is to be used (note that this array
            therefore defines the forward coordinate transformation).
            Coordinates are numbered starting from 1.
            Values may also be negative; see the `constant` parameter for details.
    @param[in] constant  An vector containing values which may be assigned to input and/or output
              coordinates instead of deriving them from other coordinate values.  If either
              of the `inperm` or `outperm` arrays contains a negative value, it is used to
              address this `constant` array (such that -1 addresses the first element, -2 addresses
              the second element, etc.)  and the value obtained is used as the corresponding
              coordinate value.
    @param[in] options  Comma-separated list of attribute assignments.

    @throw std::invalid_argument if:
    - `inperm` or `outperm` are empty
    - `inperm` or `outperm` specify a constant that is not available because `constant` has too few elements.
    */
    explicit PermMap(std::vector<int> const & inperm, 
                     std::vector<int> const & outperm,
                     std::vector<double> const & constant={},
                     std::string const & options="") :
        Mapping(reinterpret_cast<AstMapping *>(makeRawMap(inperm, outperm, constant, options)))
    {}

    virtual ~PermMap() {}

    PermMap(PermMap const &) = delete;
    PermMap(PermMap &&) = default;
    PermMap & operator=(PermMap const &) = delete;
    PermMap & operator=(PermMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<PermMap> copy() const {
        return std::static_pointer_cast<PermMap>(_copyPolymorphic());
    }

protected:
    virtual std::shared_ptr<Object> _copyPolymorphic() const {
        return _copyImpl<PermMap, AstPermMap>();
    }    

    /// Construct a PermMap from a raw AST pointer   
    explicit PermMap(AstPermMap * rawptr) :
        Mapping(reinterpret_cast<AstMapping *>(rawptr))
    {
        if (!astIsAPermMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClass() << ", which is not a PermMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    AstPermMap * makeRawMap(std::vector<int> const & inperm, 
                            std::vector<int> const & outperm,
                            std::vector<double> const & constant={},
                            std::string const & options="") {
        if (inperm.empty()) {
            throw std::invalid_argument("inperm has no elements");
        }
        if (outperm.empty()) {
            throw std::invalid_argument("outperm has no elements");
        }
        // check `constant` (since AST does not)
        checkConstant(constant.size(), inperm, "inperm");
        checkConstant(constant.size(), outperm, "outperm");

        double const * constptr = constant.size() > 0 ? constant.data() : nullptr;
        return astPermMap(inperm.size(), inperm.data(),
                          outperm.size(), outperm.data(),
                          constptr, options.c_str());
    }
};

}  // namespace ast

#endif
