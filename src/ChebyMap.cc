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
#include "astshim/ChebyMap.h"

namespace ast {

ChebyMap ChebyMap::polyTran(bool forward, double acc, double maxacc, int maxorder,
                            std::vector<double> const &lbnd, std::vector<double> const &ubnd) const {
    return ChebyMap(detail::polyTranImpl<AstChebyMap>(*this, forward, acc, maxacc, maxorder, lbnd, ubnd));
}

ChebyMap ChebyMap::polyTran(bool forward, double acc, double maxacc, int maxorder) const {
    // Note: AST directly supports specifying lbnd or ubnd as NULL for ChebyMap
    // but the code provide explicit bounds for two reasons:
    // - This simplifies the interface to polyTranImpl.
    // - This simplifies the interface to ChebyMap.polyTran by requiring
    //   either both boundaries or neither. Though admittedly this code
    //   could let polyTranImpl do the work instead of calling getDomain.
    // - When this code was written, the feature of allowing NULL for lbnd or ubnd
    //   didn't work (though I expect it to be fixed very soon).
    auto domain = getDomain(!forward);
    return ChebyMap(detail::polyTranImpl<AstChebyMap>(*this, forward, acc, maxacc, maxorder, domain.lbnd,
                                                      domain.ubnd));
}

ChebyMap::ChebyMap(AstChebyMap *map) : Mapping(reinterpret_cast<AstMapping *>(map)) {
    if (!astIsAChebyMap(getRawPtr())) {
        std::ostringstream os;
        os << "this is a " << getClassName() << ", which is not a ChebyMap";
        throw std::invalid_argument(os.str());
    }
}

ChebyDomain ChebyMap::getDomain(bool forward) const {
    int nElements = forward ? getNIn() : getNOut();
    std::vector<double> lbnd(nElements, 0.0);
    std::vector<double> ubnd(nElements, 0.0);
    astChebyDomain(getRawPtr(), static_cast<int>(forward), lbnd.data(), ubnd.data());
    assertOK();
    for (auto &val : lbnd) {
        if (val == AST__BAD) {
            throw std::runtime_error("Could not compute domain");
        }
    }
    for (auto &val : ubnd) {
        if (val == AST__BAD) {
            throw std::runtime_error("Could not compute domain");
        }
    }
    return ChebyDomain(lbnd, ubnd);
}

/// Make a raw AstChebyMap with specified forward and inverse transforms.
AstChebyMap *ChebyMap::_makeRawChebyMap(ConstArray2D const &coeff_f, ConstArray2D const &coeff_i,
                                        std::vector<double> const &lbnd_f, std::vector<double> const &ubnd_f,
                                        std::vector<double> const &lbnd_i, std::vector<double> const &ubnd_i,
                                        std::string const &options) const {
    const int nin = coeff_f.getSize<1>() - 2;
    const int ncoeff_f = coeff_f.getSize<0>();
    const int nout = coeff_i.getSize<1>() - 2;
    const int ncoeff_i = coeff_i.getSize<0>();
    const bool has_fwd = ncoeff_f > 0;
    const bool has_inv = ncoeff_i > 0;

    if (!has_fwd and !has_inv) {
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
    if (has_fwd) {
        detail::assertEqual(lbnd_f.size(), "lbnd_f size", static_cast<std::size_t>(nin),
                            "number of input axes");
        detail::assertEqual(ubnd_f.size(), "ubnd_f size", static_cast<std::size_t>(nin),
                            "number of input axes");
    }
    if (has_inv) {
        detail::assertEqual(lbnd_i.size(), "lbnd_i size", static_cast<std::size_t>(nout),
                            "number of output axes");
        detail::assertEqual(ubnd_i.size(), "ubnd_i size", static_cast<std::size_t>(nout),
                            "number of output axes");
    }

    auto result = reinterpret_cast<AstChebyMap *>(astChebyMap(nin, nout, ncoeff_f, coeff_f.getData(),
                                                              ncoeff_i, coeff_i.getData(), lbnd_f.data(),
                                                              ubnd_f.data(), lbnd_i.data(), ubnd_i.data(),
                                                              "%s", options.c_str()));
    assertOK();
    return result;
}

/// Make a raw AstChebyMap with a specified forward transform and an optional iterative inverse.
AstChebyMap *ChebyMap::_makeRawChebyMap(ConstArray2D const &coeff_f, int nout,
                                        std::vector<double> const &lbnd_f, std::vector<double> const &ubnd_f,
                                        std::string const &options) const {
    const int nin = coeff_f.getSize<1>() - 2;
    const int ncoeff_f = coeff_f.getSize<0>();
    const bool has_fwd = ncoeff_f > 0;

    if (!has_fwd) {
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
        os << "coeff_i row length " << nout + 2
           << ", which is too short; length = nout + 2 and nout must be > 0";
        throw std::invalid_argument(os.str());
    }
    detail::assertEqual(lbnd_f.size(), "lbnd_f size", static_cast<std::size_t>(nin), "number of input axes");
    detail::assertEqual(ubnd_f.size(), "ubnd_f size", static_cast<std::size_t>(nin), "number of input axes");

    return reinterpret_cast<AstChebyMap *>(astChebyMap(nin, nout, ncoeff_f, coeff_f.getData(), 0, nullptr,
                                                       lbnd_f.data(), ubnd_f.data(), nullptr, nullptr, "%s",
                                                       options.c_str()));
}

}  // namespace ast