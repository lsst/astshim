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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <vector>

#include "ndarray.h"

#include "astshim.h"

int main() {
    std::vector<double> fromVec = {1.1, 1.2, 2.1, 2.2, 3.1, 3.2};

    // Mappings can transform vectors or 2-d arrays;
    // this example uses 2-d arrays because they can be printed to stdout.
    ast::Array2D from = ast::arrayFromVector(fromVec, 2);

    auto zoomMap = ast::ZoomMap(2, 5);
    std::cout << "ref count for Mapping = " << zoomMap.getRefCount() << std::endl;
    std::cout << "zoomMap for Mapping = " << zoomMap.getZoom() << "; is inverted=" << zoomMap.isInverted()
              << std::endl;

    ast::Array2D to = ndarray::allocate(ndarray::makeVector(3, 2));
    zoomMap.tranForward(from, to);

    auto invZoomPtr = zoomMap.getInverse();
    ast::Array2D rndTrip = ndarray::allocate(ndarray::makeVector(3, 2));
    invZoomPtr->tranForward(to, rndTrip);
    std::cout << "x =" << from << std::endl;
    std::cout << "zoom(x) =" << to << std::endl;
    std::cout << "round trip =" << rndTrip << std::endl;

    std::vector<double> shift = {1.5, -0.1};
    auto shiftMap = ast::ShiftMap(shift);

    auto seriesMap = zoomMap.of(shiftMap);
    seriesMap.tranForward(from, to);
    std::cout << "\n\nx =" << from << std::endl;
    shiftMap.tranForward(from, to);
    std::cout << "shift(x) =" << to << std::endl;
    seriesMap.tranForward(from, to);
    std::cout << "zoom(shift(x) =" << to << std::endl;

    auto shiftMapCopy = seriesMap[0];
    std::cout << "\n\nx =" << from << std::endl;
    shiftMapCopy->tranForward(from, to);
    std::cout << "shiftCopy(x) =" << to << std::endl;
}
