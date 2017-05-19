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
#ifndef ASTSHIM_QUADAPPROX_H
#define ASTSHIM_QUADAPPROX_H

#include <vector>

namespace ast {
class Mapping;

/**
A quadratic approximation to a 2D Mapping.

Construct the class to compute the contained fields:

*/
class QuadApprox {
public:
    /**
    Obtain a quadratic approximation to a 2D Mapping.

    Compute the coefficients of a quadratic fit to the
    supplied Mapping over the input area specified by `lbnd` and `ubnd`.
    The Mapping must have 2 inputs, but may have any number of outputs.
    The i'th Mapping output is modeled as a quadratic function of the
    2 inputs (x,y):

    output_i = a_i_0 + a_i_1*x + a_i_2*y + a_i_3*x*y + a_i_4*x*x +
               a_i_5*y*y

    The `fit` vector is set tothe values of the co-efficients a_0_0, a_0_1, etc.

    @param[in] map  Mapping to fit.
    @param[in] lbnd  The lower bounds of a box defined within the input
       coordinate system of the Mapping. The number of elements in this
       vector should equal map.getNIn(). This
       box should specify the region over which the fit is to be
       performed.
    @param[in] ubnd  The upper bounds of the box specifying the region over
       which the fit is to be performed.
    @param[in] nx  The number of points to place along the first Mapping input. The
       first point is at
       `lbnd[0]` and the last is at `ubnd[0]"`.
       If a value less than three is supplied a value of three will be used.
    @param[in] ny   The number of points to place along the second Mapping input. The
       first point is at `lbnd[1]` and the last is at `ubnd[1]`.
       If a value less than three is supplied a value of three will be used.

    @throws std::invalid_argument if the mapping does not have 2 inputs,
        or if lbnd or ubnd do not each contain 2 elements.
    @throws std::runtime_error if the fit cannot be computed.
    */
    explicit QuadApprox(Mapping const &map, std::vector<double> const &lbnd, std::vector<double> const &ubnd,
                        int nx = 3, int ny = 3);

    QuadApprox(QuadApprox const &) = default;
    QuadApprox(QuadApprox &&) = default;
    QuadApprox &operator=(QuadApprox const &) = default;
    QuadApprox &operator=(QuadApprox &&) = default;

    /**
    A vector of coefficients of the quadratic approximation to the specified transformation.

    This vector will contain "6*NOut", elements:
    the first 6 elements hold the fit to the first Mapping output,
    the next 6 elements hold the fit to the second Mapping output, etc.
    So if the Mapping has 2 inputs and 2 outputs the quadratic approximation
    to the forward transformation is:

        X_out = fit[0] + fit[1]*X_in + fit[2]*Y_in + fit[3]*X_in*Y_in +
                fit[4]*X_in*X_in + fit[5]*Y_in*Y_in
        Y_out = fit[6] + fit[7]*X_in + fit[8]*Y_in + fit[9]*X_in*Y_in +
                fit[10]*X_in*X_in + fit[11]*Y_in*Y_in
        X_out = fit(1) + fit(2)*X_in + fit(3)*Y_in + fit(4)*X_in*Y_in +
                fit(5)*X_in*X_in + fit(6)*Y_in*Y_in
        Y_out = fit(7) + fit(8)*X_in + fit(9)*Y_in + fit(10)*X_in*Y_in +
                fit(11)*X_in*X_in + fit(12)*Y_in*Y_in
    */
    std::vector<double> fit;
    /**
    The RMS residual between the fit and the Mapping, summed over all Mapping outputs.
    */
    double rms;
};

}  // namespace ast

#endif
