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
    return PolyMap(detail::polyTranImpl<AstPolyMap>(*this, forward, acc, maxacc, maxorder, lbnd, ubnd));
}

PolyMap::PolyMap(AstPolyMap *map) : Mapping(reinterpret_cast<AstMapping *>(map)) {
    if (!astIsAPolyMap(getRawPtr())) {
        std::ostringstream os;
        os << "this is a " << getClass() << ", which is not a PolyMap";
        throw std::invalid_argument(os.str());
    }
}

/// Make a raw AstPolyMap with specified forward and inverse transforms.
AstPolyMap *PolyMap::_makeRawPolyMap(ndarray::Array<double, 2, 2> const &coeff_f,
                                     ndarray::Array<double, 2, 2> const &coeff_i,
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

    return astPolyMap(nin, nout, ncoeff_f, coeff_f.getData(), ncoeff_i, coeff_i.getData(), options.c_str());
}

/// Make a raw AstPolyMap with a specified forward transform and an optional iterative inverse.
AstPolyMap *PolyMap::_makeRawPolyMap(ndarray::Array<double, 2, 2> const &coeff_f, int nout,
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

    return astPolyMap(nin, nout, ncoeff_f, coeff_f.getData(), 0, nullptr, options.c_str());
}

}  // namespace ast