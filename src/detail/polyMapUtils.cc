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
#include "astshim/detail/polyMapUtils.h"
#include "astshim/ChebyMap.h"
#include "astshim/PolyMap.h"

namespace ast {
namespace detail {

template <class AstMapT, class MapT>
AstMapT *polyTranImpl(MapT const &mapping, bool forward, double acc, double maxacc, int maxorder,
                      std::vector<double> const &lbnd, std::vector<double> const &ubnd) {
    // desired size of lbnd and ubnd
    auto const bndSize = static_cast<unsigned int>(forward ? mapping.getNOut() : mapping.getNIn());

    if (lbnd.size() != bndSize) {
        std::ostringstream os;
        os << "lbnd.size() = " << lbnd.size() << " != " << bndSize << " = "
           << (forward ? "getNOut()" : "getNIn()");
        throw std::invalid_argument(os.str());
    }
    if (ubnd.size() != bndSize) {
        std::ostringstream os;
        os << "ubnd.size() = " << ubnd.size() << " != " << bndSize << " = "
           << (forward ? "getNOut()" : "getNIn()");
        throw std::invalid_argument(os.str());
    }

    void *outRawMap = astPolyTran(mapping.getRawPtr(), static_cast<int>(forward), acc, maxacc, maxorder,
                                  lbnd.data(), ubnd.data());
    // Failure should result in a null pointer, so calling assertOK is unlikely to do anything,
    // but better to be sure and than risk missing an uncaught error.
    assertOK(reinterpret_cast<AstObject *>(outRawMap));
    if (!outRawMap) {
        throw std::runtime_error("Could not compute an inverse mapping");
    }
    return reinterpret_cast<AstMapT *>(outRawMap);
}

// Explicit instantiations
template AstChebyMap *polyTranImpl<AstChebyMap>(ChebyMap const &, bool, double, double, int,
                                                std::vector<double> const &, std::vector<double> const &);
template AstPolyMap *polyTranImpl<AstPolyMap>(PolyMap const &, bool, double, double, int,
                                              std::vector<double> const &, std::vector<double> const &);

}  // namespace detail
}  // namespace ast