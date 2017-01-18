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
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Mapping.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"

namespace ast {

SeriesMap Mapping::of(Mapping const & first) const {
    return SeriesMap(first, *this);
}

ParallelMap Mapping::over(Mapping const & first) const {
    return ParallelMap(first, *this);
}

void Mapping::_tran(
    Array2D const & from,
    bool doForward,
    Array2D & to
) const {
    int const nFromAxes = doForward ? getNin()  : getNout();
    int const nToAxes   = doForward ? getNout() : getNin();
    detail::assertEqual(from.getSize<1>(), "from.size[1]", static_cast<std::size_t>(nFromAxes), "from coords");
    detail::assertEqual(to.getSize<1>(), "to.size[1]", static_cast<std::size_t>(nToAxes), "to coords");
    detail::assertEqual(from.getSize<0>(), "from.size[1]", to.getSize<0>(), "to.size[1]");
    int const nPts = from.getSize<0>();
    // astTranN uses fortran ordering x0, x1, x2, ..., y0, y1, y2, ..., ... so transpose in and out
    Array2D fromT = ndarray::copy(from.transpose());
    Array2D toT = ndarray::allocate(ndarray::makeVector(nToAxes, nPts));
    astTranN(getRawPtr(), nPts, nFromAxes, nPts, fromT.getData(),
             static_cast<int>(doForward), nToAxes, nPts, toT.getData());
    assertOK();
    to.transpose() = toT;
    detail::astBadToNan(to);
}

void Mapping::_tranGrid(
    PointI const & lbnd,
    PointI const & ubnd,
    double tol,
    int maxpix,
    bool doForward,
    Array2D & to
) const {
    int const nFromAxes = doForward ? getNin()  : getNout();
    int const nToAxes   = doForward ? getNout() : getNin();
    detail::assertEqual(lbnd.size(), "lbnd.size", static_cast<std::size_t>(nFromAxes), "from coords");
    detail::assertEqual(ubnd.size(), "ubnd.size", static_cast<std::size_t>(nFromAxes), "from coords");
    detail::assertEqual(to.getSize<1>(), "to.size[0]", static_cast<std::size_t>(nToAxes), "to coords");
    int const nPts = to.getSize<0>();
    Array2D toT = ndarray::allocate(ndarray::makeVector(nToAxes, nPts));
    astTranGrid(getRawPtr(), nFromAxes, lbnd.data(), ubnd.data(),
                tol, maxpix, static_cast<int>(doForward), nToAxes, nPts, toT.getData());
    assertOK();
    to.transpose() = toT;
    detail::astBadToNan(to);
}

}  // namespace ast
