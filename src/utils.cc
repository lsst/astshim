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

#include "astshim/utils.h"

namespace ast {

namespace {

/*
 * Compute number of coefficients from polynomial order.
 *
 * @throws std::invalid_argument if order < 0
 */
int nCoeffsFromOrder(int order) {
    if (order < 0) {
        std::ostringstream os;
        os << "order=" << order << " invalid: must be >= 0";
        throw std::invalid_argument(os.str());
    }
    return (order + 1) * (order + 2) / 2;
}

/**
 * Compute polynomial order from the number of coefficients
 *
 * Only certain values of nCoeffs are acceptable, including:
 * nCoeffs  order
 *      1       0
 *      3       1
 *      6       2
 *     10       3
 *     15       4
 *    ...
 *
 * @throws std::invalid_argument if nCoeffs is invalid
 */
int orderFromNCoeffs(int nCoeffs) {
    int order =
            static_cast<int>(0.5 + ((-3.0 + (std::sqrt(1.0 + (8.0 * static_cast<double>(nCoeffs))))) / 2.0));
    if (nCoeffs != nCoeffsFromOrder(order)) {
        std::ostringstream os;
        os << "nCoeffs=" << nCoeffs << " invalid: order is not an integer";
        throw std::invalid_argument(os.str());
    }
    return order;
}

}  // namespace

ndarray::Array<double, 2, 2> makePolynomialCoeffs11(std::vector<double> const &coeffs) {
    int const nCoeffs = coeffs.size();
    ndarray::Array<double, 2, 2> const astCoeffs = ndarray::allocate(ndarray::makeVector(nCoeffs, 3));
    for (size_t i = 0; i < coeffs.size(); ++i) {
        astCoeffs[i][0] = coeffs[i];
        astCoeffs[i][1] = 1;
        astCoeffs[i][2] = i;
    }
    return astCoeffs;
}

ndarray::Array<double, 2, 2> makePolynomialCoeffs21(std::vector<double> const &coeffs) {
    int const nCoeffs = coeffs.size();
    int order = orderFromNCoeffs(nCoeffs);
    ndarray::Array<double, 2, 2> const astCoeffs = ndarray::allocate(ndarray::makeVector(nCoeffs, 4));
    int i = 0;
    for (int nx = 0; nx < order + 1; ++nx) {
        for (int ny = 0; ny < order + 1 - nx; ++ny, ++i) {
            astCoeffs[i][0] = coeffs[i];
            astCoeffs[i][1] = 1;
            astCoeffs[i][2] = nx;
            astCoeffs[i][3] = ny;
        }
    }
    return astCoeffs;
}

ndarray::Array<double, 2, 2> makePolynomialCoeffs22(std::vector<double> const &xCoeffs,
                                                    std::vector<double> const &yCoeffs) {
    if (xCoeffs.size() != yCoeffs.size()) {
        std::ostringstream os;
        os << "xCoeffs.size() = " << xCoeffs.size() << " != yCoeffs.size() = " << yCoeffs.size();
        throw std::invalid_argument(os.str());
    }
    int const nxCoeffs = xCoeffs.size();
    int order = orderFromNCoeffs(nxCoeffs);
    ndarray::Array<double, 2, 2> const astCoeffs = ndarray::allocate(ndarray::makeVector(nxCoeffs * 2, 4));
    int i = 0;
    for (int nx = 0; nx < order + 1; ++nx) {
        for (int ny = 0; ny < order + 1 - nx; ++ny, ++i) {
            int const astInd1 = i * 2;
            int const astInd2 = astInd1 + 1;
            astCoeffs[astInd1][0] = xCoeffs[astInd1];
            astCoeffs[astInd1][1] = 1;
            astCoeffs[astInd1][2] = nx;
            astCoeffs[astInd1][3] = ny;
            astCoeffs[astInd2][0] = yCoeffs[i];
            astCoeffs[astInd2][1] = 2;
            astCoeffs[astInd2][2] = nx;
            astCoeffs[astInd2][3] = ny;
        }
    }
    return astCoeffs;
}

}  // namespace ast
