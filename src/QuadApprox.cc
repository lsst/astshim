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

#include <vector>

#include "astshim/detail/utils.h"
#include "astshim/Mapping.h"
#include "astshim/QuadApprox.h"

namespace ast {

QuadApprox::QuadApprox(Mapping const& map, std::vector<double> const& lbnd, std::vector<double> const& ubnd,
                       int nx, int ny)
        : fit(6 * map.getNOut()), rms(0) {
    int const nIn = map.getNIn();
    detail::assertEqual(nIn, "map.getNIn()", 2, "required nIn");
    detail::assertEqual(lbnd.size(), "lbnd.size", static_cast<std::size_t>(nIn), "nIn");
    detail::assertEqual(ubnd.size(), "ubnd.size", static_cast<std::size_t>(nIn), "nIn");
    fit.reserve(6 * map.getNOut());
    bool isok = astQuadApprox(map.getRawPtr(), lbnd.data(), ubnd.data(), nx, ny, fit.data(), &rms);
    assertOK();
    if (!isok) {
        throw std::runtime_error("Failed to fit a quadratic approximation");
    }
}

}  // namespace ast