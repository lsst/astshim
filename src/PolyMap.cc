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
#include <sstream>
#include <stdexcept>

#include "astshim/detail/polyMapUtils.h"
#include "astshim/PolyMap.h"

namespace ast {

PolyMap PolyMap::polyTran(bool forward, double acc, double maxacc, int maxorder,
                          std::vector<double> const &lbnd, std::vector<double> const &ubnd) const {
    // If there is an iterative inverse then reject attempts to fit the other direction.
    // AST catches the case that there are no inverse coefficients,
    // but I prefer to also raise if there are inverse coefficients because
    // the iterative inverse cannot match the inverse coefficients, except in the most trivial cases,
    // and the inverse coefficients are used to fit the forward direction,
    // so the results are likely to be surprising
    if (getIterInverse()) {
        if (forward != isInverted()) {
            if (forward) {
                throw std::invalid_argument("Cannot fit forward transform when inverse is iterative");
            } else {
                throw std::invalid_argument("Cannot fit inverse transform when forward is iterative");
            }
        }
    }
    return PolyMap(detail::polyTranImpl<AstPolyMap>(*this, forward, acc, maxacc, maxorder, lbnd, ubnd));
}

PolyMap::PolyMap(AstPolyMap *map) : Mapping(reinterpret_cast<AstMapping *>(map)) {
    if (!astIsAPolyMap(getRawPtr())) {
        std::ostringstream os;
        os << "this is a " << getClassName() << ", which is not a PolyMap";
        throw std::invalid_argument(os.str());
    }
}

/// Make a raw AstPolyMap with specified forward and inverse transforms.
AstPolyMap *PolyMap::_makeRawPolyMap(ConstArray2D const &coeff_f, ConstArray2D const &coeff_i,
                                     std::string const &options) const {
    const int nin = coeff_f.getSize<1>() - 2;
    const int ncoeff_f = coeff_f.getSize<0>();
    const int nout = coeff_i.getSize<1>() - 2;
    const int ncoeff_i = coeff_i.getSize<0>();

    if ((ncoeff_f == 0) && (ncoeff_i == 0)) {
        throw std::invalid_argument(
                "Must specify forward or inverse transform (coeff_f and coeff_i both empty)");
    }
    if (nin <= 0) {
        std::ostringstream os;
        os << "coeff_f row length = " << nin + 2
           << ", which is too short; length = nin + 2 and nin must be > 0";
        throw std::invalid_argument(os.str());
    }
    if (nout <= 0) {
        std::ostringstream os;
        os << "coeff_i row length " << nout + 2
           << ", which is too short; length = nout + 2 and nout must be > 0";
        throw std::invalid_argument(os.str());
    }

    auto result = astPolyMap(nin, nout, ncoeff_f, coeff_f.getData(), ncoeff_i, coeff_i.getData(), "%s",
                             options.c_str());
    assertOK();
    return result;
}

/// Make a raw AstPolyMap with a specified forward transform and an optional iterative inverse.
AstPolyMap *PolyMap::_makeRawPolyMap(ConstArray2D const &coeff_f, int nout,
                                     std::string const &options) const {
    const int nin = coeff_f.getSize<1>() - 2;
    const int ncoeff_f = coeff_f.getSize<0>();
    if (ncoeff_f <= 0) {
        throw std::invalid_argument("Must specify forward transform (coeff_f is empty)");
    }
    if (nin <= 0) {
        std::ostringstream os;
        os << "coeff_f row length = " << nin + 2
           << ", which is too short; length = nin + 2 and nin must be > 0";
        throw std::invalid_argument(os.str());
    }
    if (nout <= 0) {
        std::ostringstream os;
        os << "nout = " << nout << " <0 =";
        throw std::invalid_argument(os.str());
    }

    auto result = astPolyMap(nin, nout, ncoeff_f, coeff_f.getData(), 0, nullptr, "%s", options.c_str());
    assertOK();
    return result;
}

}  // namespace ast