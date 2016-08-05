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

#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/detail.h"

#include "astshim/MapBox.h"
#include "astshim/Mapping.h"

namespace ast {

MapBox::MapBox(Mapping const & map,
                std::vector<double> const & lbnd,
                std::vector<double> const & ubnd,
                int minOutCoord, int maxOutCoord) :
    lbndIn(lbnd),
    ubndIn(ubnd),
    minOutCoord(minOutCoord),
    maxOutCoord(maxOutCoord),
    lbndOut(),
    ubndOut(),
    xl(),
    xu()
{
    _compute(map, lbnd, ubnd, minOutCoord, maxOutCoord);
}

void MapBox::_compute(Mapping const & map,
              std::vector<double> const & lbnd,
              std::vector<double> const & ubnd,
              int minOutCoord, int maxOutCoord) {
    int const nin = map.getNin();
    int const nout = map.getNout();
    detail::assertEqual(lbnd.size(), "lbnd.size()", nin, "Nin");
    detail::assertEqual(ubnd.size(), "ubnd.size()", nin, "Nin");
    if (maxOutCoord == 0) {
        maxOutCoord = nout;
    } else {
       if (maxOutCoord > nout) {
           std::ostringstream os;
           os << "maxOutCoord = " << maxOutCoord << " not in range [1, " << nout
                << "], or 0 for all remaining";
           throw std::invalid_argument(os.str());
       }
    }
    if ((minOutCoord < 0) || (minOutCoord > maxOutCoord)) {
           std::ostringstream os;
           os << "minOutCoord = " << minOutCoord << " not in range [1, " << maxOutCoord << "]";
           throw std::invalid_argument(os.str());
    }
    int const npoints = 1 + maxOutCoord - minOutCoord;
    lbndOut.reserve(npoints);
    ubndOut.reserve(npoints);
    xl = ndarray::allocate(ndarray::makeVector(npoints, nout));
    xu = ndarray::allocate(ndarray::makeVector(npoints, nout));
    bool const forward = true;
    double lbndOut_i;
    double ubndOut_i;
    for (int i = 0, outcoord = minOutCoord; outcoord <= maxOutCoord; ++i, ++outcoord) {
        auto xlrow = xl[i];
        auto xurow = xu[i];
        astMapBox(map.getRawPtr(), lbnd.data(), ubnd.data(),
                  forward, outcoord, &lbndOut_i, &ubndOut_i,
                  xlrow.getData(), xurow.getData());
        assertOK();
        lbndOut.push_back(lbndOut_i);
        ubndOut.push_back(ubndOut_i);
    }

    // convert AST__BAD to nan
    detail::astBadToNan(lbndOut);
    detail::astBadToNan(ubndOut);
    detail::astBadToNan(xl);
    detail::astBadToNan(xu);
}

}  // namespace ast