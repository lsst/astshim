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
#include <memory>
#include <algorithm>  // for std::max
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Mapping.h"
#include "astshim/PermMap.h"

namespace {

/**
Throw std::invalid_argument if a permutation array calls for a constant that is not available

@param[in] numConst  Size of constant array
@param[in] perm  Permutation vector (inperm or outperm)
@param[in] name  Name of permutation vector, to use in reporting a problem
*/
void checkConstant(int numConst, std::vector<int> const& perm, std::string const& name) {
    int maxConst = 0;
    for (int const& innum : perm) {
        maxConst = std::max(maxConst, -innum);
    }
    if (maxConst > numConst) {
        std::ostringstream os;
        os << name << " specifies max constant number (min negative number) " << maxConst << ", but only "
           << numConst << " constants are available";
        throw std::invalid_argument(os.str());
    }
}

}  // anonymous namespace

namespace ast {

AstPermMap* PermMap::makeRawMap(std::vector<int> const& inperm, std::vector<int> const& outperm,
                                std::vector<double> const& constant, std::string const& options) {
    if (inperm.empty()) {
        throw std::invalid_argument("inperm has no elements");
    }
    if (outperm.empty()) {
        throw std::invalid_argument("outperm has no elements");
    }
    // check `constant` (since AST does not)
    checkConstant(constant.size(), inperm, "inperm");
    checkConstant(constant.size(), outperm, "outperm");

    double const* constptr = constant.size() > 0 ? constant.data() : nullptr;
    auto result = astPermMap(inperm.size(), inperm.data(), outperm.size(), outperm.data(), constptr, "%s",
                             options.c_str());
    assertOK();
    return result;
}

}  // namespace ast
