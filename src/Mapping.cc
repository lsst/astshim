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
#include <memory>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Frame.h"
#include "astshim/Mapping.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"

namespace ast {

SeriesMap Mapping::of(Mapping const &first) const { return SeriesMap(first, *this); }

ParallelMap Mapping::over(Mapping const &first) const { return ParallelMap(first, *this); }

std::shared_ptr<Mapping> Mapping::getInverse() const {
    auto rawCopy = reinterpret_cast<AstMapping *>(astCopy(getRawPtr()));
    astInvert(rawCopy);
    assertOK(reinterpret_cast<AstObject *>(rawCopy));
    // use false because the pointer has already been copied
    return Object::fromAstObject<Mapping>(reinterpret_cast<AstObject *>(rawCopy), false);
}

Array2D Mapping::linearApprox(PointD const &lbnd, PointD const &ubnd, double tol) const {
    int const nIn = getNin();
    int const nOut = getNout();
    detail::assertEqual(lbnd.size(), "lbnd.size", static_cast<std::size_t>(nIn), "nIn");
    detail::assertEqual(ubnd.size(), "ubnd.size", static_cast<std::size_t>(nIn), "nIn");
    Array2D fit = ndarray::allocate(ndarray::makeVector(1 + nIn, nOut));
    int isOK = astLinearApprox(getRawPtr(), lbnd.data(), ubnd.data(), tol, fit.getData());
    if (!isOK) {
        throw std::runtime_error("Mapping not sufficiently linear");
    }
    assertOK();
    return fit;
}

template <typename Class>
std::shared_ptr<Class> Mapping::_decompose(int i, bool copy) const {
    if ((i < 0) || (i > 1)) {
        std::ostringstream os;
        os << "i =" << i << "; must be 0 or 1";
        throw std::invalid_argument(os.str());
    }
    // Report pre-existing problems now so our later test for "not a compound object" is accurate
    assertOK();

    AstMapping *rawMap1;
    AstMapping *rawMap2;
    int series, invert1, invert2;
    astDecompose(getRawPtr(), &rawMap1, &rawMap2, &series, &invert1, &invert2);

    if (!rawMap2) {
        // Not a compound object; free rawMap1 (rawMap2 is null, so no need to free it) and throw an exception
        astAnnul(reinterpret_cast<AstObject *>(rawMap1));
        std::ostringstream os;
        os << "This " << getClass() << " is not a compound object";
        throw std::runtime_error(os.str());
    }

    // Make a deep copy of the returned object and free the shallow copies
    AstMapping *retRawMap;
    int invert;
    if (i == 0) {
        retRawMap = reinterpret_cast<AstMapping *>(astCopy(reinterpret_cast<AstObject *>(rawMap1)));
        invert = invert1;
    } else {
        retRawMap = reinterpret_cast<AstMapping *>(astCopy(reinterpret_cast<AstObject *>(rawMap2)));
        invert = invert2;
    }
    astAnnul(reinterpret_cast<AstObject *>(rawMap1));
    astAnnul(reinterpret_cast<AstObject *>(rawMap2));

    // If the mapping's internal invert flag does not match the value used when the CmpMap was made
    // then invert the mapping. Note that it is not possible to create such objects in astshim
    // but it is possible to read in objects created by other software.
    if (invert != astGetI(retRawMap, "Invert")) {
        astInvert(retRawMap);
    }

    return Object::fromAstObject<Class>(reinterpret_cast<AstObject *>(retRawMap), copy);
}

void Mapping::_tran(ConstArray2D const &from, bool doForward, Array2D const &to) const {
    int const nFromAxes = doForward ? getNin() : getNout();
    int const nToAxes = doForward ? getNout() : getNin();
    detail::assertEqual(from.getSize<1>(), "from.size[1]", static_cast<std::size_t>(nFromAxes),
                        "from coords");
    detail::assertEqual(to.getSize<1>(), "to.size[1]", static_cast<std::size_t>(nToAxes), "to coords");
    detail::assertEqual(from.getSize<0>(), "from.size[1]", to.getSize<0>(), "to.size[1]");
    int const nPts = from.getSize<0>();
    // astTranN uses fortran ordering x0, x1, x2, ..., y0, y1, y2, ..., ... so transpose in and out
    Array2D fromT = ndarray::copy(from.transpose());
    Array2D toT = ndarray::allocate(ndarray::makeVector(nToAxes, nPts));
    astTranN(getRawPtr(), nPts, nFromAxes, nPts, fromT.getData(), static_cast<int>(doForward), nToAxes, nPts,
             toT.getData());
    assertOK();
    to.transpose() = toT;
    detail::astBadToNan(to);
}

void Mapping::_tranGrid(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix, bool doForward,
                        Array2D const &to) const {
    int const nFromAxes = doForward ? getNin() : getNout();
    int const nToAxes = doForward ? getNout() : getNin();
    detail::assertEqual(lbnd.size(), "lbnd.size", static_cast<std::size_t>(nFromAxes), "from coords");
    detail::assertEqual(ubnd.size(), "ubnd.size", static_cast<std::size_t>(nFromAxes), "from coords");
    detail::assertEqual(to.getSize<1>(), "to.size[0]", static_cast<std::size_t>(nToAxes), "to coords");
    int const nPts = to.getSize<0>();
    Array2D toT = ndarray::allocate(ndarray::makeVector(nToAxes, nPts));
    astTranGrid(getRawPtr(), nFromAxes, lbnd.data(), ubnd.data(), tol, maxpix, static_cast<int>(doForward),
                nToAxes, nPts, toT.getData());
    assertOK();
    to.transpose() = toT;
    detail::astBadToNan(to);
}

// Explicit instantiations
template std::shared_ptr<Frame> Mapping::_decompose(int i, bool) const;
template std::shared_ptr<Mapping> Mapping::_decompose(int i, bool) const;

}  // namespace ast
