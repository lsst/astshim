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
#ifndef ASTSHIM_DETAIL_POLYMAPUTILS_H
#define ASTSHIM_DETAIL_POLYMAPUTILS_H

#include <stdexcept>

#include "astshim/base.h"

namespace ast {
namespace detail {

/**
Call astPolyTran to set (or replace) one direction of a polynomial transform
with a fit based on the other direction.

@tparam AstMapT  AST mapping class: one of AstChebyMap or AstPolyMap
@tparam MapT  Corresponding astshim class: one of ast::ChebyMap or ast::PolyMap;
    this template parameter is second because it can always be deduced.

@param[in] mapping
@param[in] forward  If true the forward transformation is replaced.
                Otherwise the inverse transformation is replaced.
@param[in] acc  The target accuracy, expressed as a geodesic distance within
                the ChebyMap's input space (if `forward` is false)
                or output space (if `forward` is true).
@param[in] maxacc  The maximum allowed accuracy for an acceptable polynomial,
                expressed as a geodesic distance within the ChebyMap's input space
                (if `forward` is false) or output space (if `forward` is true).
@param[in] maxorder  The maximum allowed polynomial order. This is one more than the
                maximum power of either input axis. So for instance, a value of
                3 refers to a quadratic polynomial.
                Note, cross terms with total powers greater than or equal to `maxorder`
                are not inlcuded in the fit. So the maximum number of terms in
                each of the fitted polynomials is `maxorder*(maxorder + 1)/2.`
@param[in] lbnd  A vector holding the lower bounds of a rectangular region within
                the ChebyMap's input space (if `forward` is false)
                or output space (if `forward` is true).
                The new polynomial will be evaluated over this rectangle. The length
                should equal getNIn() or getNOut(), depending on `forward`.
@param[in] ubnd  A vector holding the upper bounds of a rectangular region within
                the ChebyMap's input space (if `forward` is false)
                or output space (if `forward` is true).
                The new polynomial will be evaluated over this rectangle. The length
                should equal getNIn() or getNOut(), depending on `forward`.

@throws std::invalid_argument if the size of `lbnd` or `ubnd` does not match getNIn() (if `forward` false)
                or getNOut() (if `forward` true).
*/
template <class AstMapT, class MapT>
AstMapT *polyTranImpl(MapT const &mapping, bool forward, double acc, double maxacc, int maxorder,
                      std::vector<double> const &lbnd, std::vector<double> const &ubnd);

}  // namespace detail
}  // namespace ast

#endif