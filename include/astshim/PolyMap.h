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
#ifndef ASTSHIM_POLYMAP_H
#define ASTSHIM_POLYMAP_H

#include <algorithm>
#include <memory>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
PolyMap is a @ref Mapping which performs a general polynomial transformation.
Each output coordinate is a polynomial function of all the input coordinates. The coefficients
are specified separately for each output coordinate. The forward and inverse transformations
are defined independantly by separate sets of coefficients. If no inverse transformation is supplied,
an iterative method can be used to evaluate the inverse based only on the forward transformation.

### Attributes

- @ref PolyMap_IterInverse "IterInverse": provide an iterative inverse transformation?
- @ref PolyMap_NiterInverse "NiterInverse": maximum number of iterations for iterative inverse.
- @ref PolyMap_TolInverse "TolInverse": target relative error for iterative inverse.
*/
class PolyMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref PolyMap with specified forward and/or inverse transforms.

    The two sets of coefficients are independent of each other: the inverse transform
    need not undo the forward transform.

    @param[in] coeff_f  A `(2 + nin) x ncoeff_f` @ref PolyMap_CoefficientMatrices "matrix of coefficients"
        describing the forward transformation. If `coeff_f` is empty then no forward transformation
        is provided.
    @param[in] coeff_i  A (2 + nout) x ncoeff_i` matrix of coefficients
        for the inverse transformation. If coeff_i is empty then no inverse transformation is provided
        unless you specify suitable options to request an iterative inverse; see the
        @ref PolyMap(ndarray::Array<double, 2, 2> const &, int, std::string const &) "other constructor"
        for details.
    @param[in] options  Comma-separated list of attribute assignments.

    @throw std::invalid_argument if neither transform is specified (coeff_f and coeff_i are both empty)

    @anchor PolyMap_CoefficientMatrices Coefficient Matrices

    The coefficients describing a forward transformation are specified as an `ncoeff_f x (2 + nin)`
    matrix, where `ncoeff_f` is the number of coefficients for the forward
    direction, and each coefficient is described by `2 + nin` contiguous values, as follows:
    - The first element is the coefficient value.
    - The next element is the integer index of the @ref PolyMap output
        which uses the coefficient within its defining polynomial
        (the first output has index 1).
    - The remaining elements give the integer power to use with each corresponding input coordinate
        value, or 0 to ignore that input coordinate. Powers must not be negative and floating point
        values are rounded to the nearest integer.

    For instance, if the @ref PolyMap has 3 inputs and 2 outputs, each row consisting
    of 5 elements, and the row `(1.2, 2, 6, 3, 0)` describes a coefficient
    that increments output 2 as follows:
        `out2 += 1.2 * in1^6 * in2^3 * in3^0`
    and the row `(-1.5, 1, 0, 0, 0)` describes a coefficient that increments
    output 2 with a constant value of -1.5 (since all powers are 0):
        `out1 += -1.5 * in1^0 * in2^0 * in3^0`
    Each final output coordinate value is the sum of the terms described
    by the `ncoeff_f` columns in the supplied array. If no coefficients increment
    a given output then it will be set to 0.

    The coefficients describing the inverse transformation work the same way,
    but of course the matrix is of size `ncoeff_i x (2 + nout)`, where `ncoeff_i` is the
    number of coefficients for the inverse transformation.
    */
    explicit PolyMap(ndarray::Array<double, 2, 2> const &coeff_f, ndarray::Array<double, 2, 2> const &coeff_i,
                     std::string const &options = "IterInverse=0")
            : Mapping(reinterpret_cast<AstMapping *>(_makeRawPolyMap(coeff_f, coeff_i, options))) {}

    /**
    Construct a @ref PolyMap with only the forward transform specified.

    If the polynomial is invertible and you want an inverse you have two choices: either
    specify suitable options to request an iterative inverse, or call polyTran to fit
    an inverse. Both have advantages:
    - The iterative inverse should provide valid values even if multiple choices exist,
        and nan if no valid choice exists, whereas polyTran will raise an exception
        if a single-valued inverse cannot be found over the specified range.
    - The iterative inverse has no range restriction, whereas polyTran produces an inverse
        that is valid over a specified range.
    - The polyTran inverse is more efficient to compute.

    @param[in] coeff_f  A `(2 + nin) x ncoeff_f` @ref PolyMap_CoefficientMatrices "matrix of coefficients"
        describing the forward transformation. If `coeff_f` is empty then no forward transformation
        is provided.
    @param[in] nout  Number of output coordinates.
    @param[in] options  Comma-separated list of attribute assignments. Useful attributes include:
        @ref PolyMap_IterInverse "IterInverse", @ref PolyMap_NiterInverse "NiterInverse" and
        @ref PolyMap_TolInverse "TolInverse".

    @throw std::invalid_argument if the forward transform is not specified (coeff_f is empty)
    */
    explicit PolyMap(ndarray::Array<double, 2, 2> const &coeff_f, int nout,
                     std::string const &options = "IterInverse=0")
            : Mapping(reinterpret_cast<AstMapping *>(_makeRawPolyMap(coeff_f, nout, options))) {}

    virtual ~PolyMap() {}

    PolyMap(PolyMap const &) = delete;
    PolyMap(PolyMap &&) = default;
    PolyMap &operator=(PolyMap const &) = delete;
    PolyMap &operator=(PolyMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<PolyMap> copy() const { return std::static_pointer_cast<PolyMap>(_copyPolymorphic()); }

    /// Get @ref PolyMap_IterInverse "IterInverse": does this provide an iterative inverse transformation?
    bool getIterInverse() const { return getB("IterInverse"); }

    /// Get @ref PolyMap_NiterInverse "NiterInverse": maximum number of iterations for iterative inverse.
    int getNiterInverse() const { return getI("NiterInverse"); }

    /// Get @ref PolyMap_TolInverse "TolInverse": target relative error for iterative inverse.
    double getTolInverse() const { return getD("TolInverse"); }

    /**
    This function creates a new @ref PolyMap which is a copy of this one,
    in which a specified transformation (forward or inverse)
    has been replaced by a new polynomial transformation. The
    coefficients of the new transformation are estimated by sampling
    the other transformation and performing a least squares polynomial
    fit in the opposite direction to the sampled positions and values.

    This method can only be used on (1-input,1-output) or (2-input, 2-output)
    @ref PolyMap "PolyMaps".

    The transformation to create is specified by the `forward` parameter.
    In what follows "X" refers to the inputs of the @ref PolyMap, and "Y" to
    the outputs of the @ref PolyMap. The forward transformation transforms
    input values (X) into output values (Y), and the inverse transformation
    transforms output values (Y) into input values (X). Within a @ref PolyMap,
    each transformation is represented by an independent set of
    polynomials, P_f or P_i: Y=P_f(X) for the forward transformation and
    X=P_i(Y) for the inverse transformation.

    The `forward` parameter specifies the transformation to be replaced.
    If it is false, a new forward transformation is created
    by first finding the input values (X) using the inverse transformation
    (which must be available) at a regular grid of points (Y) covering a
    rectangular region of the @ref PolyMap's output space. The coefficients of
    the required forward polynomial, Y=P_f(X), are chosen in order to
    minimise the sum of the squared residuals between the sampled values
    of Y and P_f(X).

    If `forward` is false (probably the most likely case),
    a new inverse transformation is created by
    first finding the output values (Y) using the forward transformation
    (which must be available) at a regular grid of points (X) covering a
    rectangular region of the @ref PolyMap's input space. The coefficients of
    the required inverse polynomial, X=P_i(Y), are chosen in order to
    minimise the sum of the squared residuals between the sampled values
    of X and P_i(Y).

    This fitting process is performed repeatedly with increasing
    polynomial orders (starting with linear) until the target
    accuracy is achieved, or a specified maximum order is reached. If
    the target accuracy cannot be achieved even with this maximum-order
    polynomial, the best fitting maximum-order polynomial is returned so
    long as its accuracy is better than
    "maxacc". If it is not, an error is reported.

    @param[in] forward  If true the forward transformation is replaced.
                    Otherwise the inverse transformation is replaced.
    @param[in] acc  The target accuracy, expressed as a geodesic distance within
                    the PolyMap's input space (if `forward` is true)
                    or output space (if `forward` is false).
    @param[in] maxacc  The maximum allowed accuracy for an acceptable polynomial,
                    expressed as a geodesic distance within the PolyMap's input space
                    (if `forward` is true) or output space (if `forward` is false).
    @param[in] maxorder  The maximum allowed polynomial order. This is one more than the
                    maximum power of either input axis. So for instance, a value of
                    3 refers to a quadratic polynomial.
                    Note, cross terms with total powers greater than or equal to `maxorder`
                    are not inlcuded in the fit. So the maximum number of terms in
                    each of the fitted polynomials is `maxorder*(maxorder + 1)/2.`
    @param[in] lbnd  A vector holding the lower bounds of a rectangular region within
                    the PolyMap's input space (if `forward` is true)
                    or output space (if `forward` is false).
                    The new polynomial will be evaluated over this rectangle. The length
                    should equal getNin() or getNout(), depending on `forward`.
    @param[in] ubnd  A vector holding the upper bounds of a rectangular region within
                    the PolyMap's input space (if `forward` is true)
                    or output space (if `forward` is false).
                    The new polynomial will be evaluated over this rectangle. The length
                    should equal getNin() or getNout(), depending on `forward`.

    @throw std::invalid_argument if lbnd.size() or ubnd.size() does not match getNin()/getNout()
                    if `forward` is true/false.
    */
    PolyMap polyTran(bool forward, double acc, double maxacc, int maxorder, std::vector<double> const &lbnd,
                     std::vector<double> const &ubnd) const;

protected:
    virtual std::shared_ptr<Object> _copyPolymorphic() const override {
        return _copyImpl<PolyMap, AstPolyMap>();
    }

    /// Construct a PolyMap from an raw AST pointer
    PolyMap(AstPolyMap *map);

private:
    /// Make a raw AstPolyMap with specified forward and inverse transforms.
    AstPolyMap *_makeRawPolyMap(ndarray::Array<double, 2, 2> const &coeff_f,
                                ndarray::Array<double, 2, 2> const &coeff_i,
                                std::string const &options = "") const;

    /// Make a raw AstPolyMap with a specified forward transform and an optional iterative inverse.
    AstPolyMap *_makeRawPolyMap(ndarray::Array<double, 2, 2> const &coeff_f, int nout,
                                std::string const &options = "") const;
};

}  // namespace ast

#endif
