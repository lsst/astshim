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
#include "astshim/detail/utils.h"

namespace ast {
namespace detail {

void astBadToNan(ast::Array2D const &arr) {
    for (auto i = arr.begin(); i != arr.end(); ++i) {
        for (auto j = i->begin(); j != i->end(); ++j) {
            if (*j == AST__BAD) {
                *j = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

std::string getClassName(AstObject const *rawObj) {
    std::string name = astGetC(rawObj, "Class");
    assertOK();
    if (name != "CmpMap") {
        return name;
    }
    bool series = isSeries(reinterpret_cast<AstCmpMap const *>(rawObj));
    return series ? "SeriesMap" : "ParallelMap";
}

bool isSeries(AstCmpMap const *cmpMap) {
    AstMapping *rawMap1;
    AstMapping *rawMap2;
    int series, invert1, invert2;
    astDecompose(cmpMap, &rawMap1, &rawMap2, &series, &invert1, &invert2);
    astAnnul(rawMap1);
    astAnnul(rawMap2);
    assertOK();
    return series;
}

}  // namespace detail
}  // namespace ast