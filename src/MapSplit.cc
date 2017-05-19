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
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Object.h"
#include "astshim/Mapping.h"
#include "astshim/MapSplit.h"

namespace ast {

MapSplit::MapSplit(Mapping const &map, std::vector<int> const &in) {
    std::vector<int> locOut;
    locOut.reserve(map.getNOut());  // the max # of elements astMapSplit may set
    AstMapping *rawSplitMap;
    astMapSplit(map.getRawPtr(), in.size(), in.data(), locOut.data(), &rawSplitMap);
    assertOK();
    if (!rawSplitMap) {
        throw std::runtime_error("Could not split the map");
    }
    splitMap = Object::fromAstObject<Mapping>(reinterpret_cast<AstObject *>(rawSplitMap), false);
    origIn = in;
    // copy the splitMap->getNOut() elements of `locOut` that contain data to `origOut`
    const int newNOut = splitMap->getNOut();
    for (int i = 0; i < newNOut; ++i) {
        origOut.push_back(locOut[i]);
    }
}

}  // namespace ast
