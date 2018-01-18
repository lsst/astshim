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
#ifndef ASTSHIM_CHEBYMAP_H
#define ASTSHIM_CHEBYMAP_H

#include <algorithm>
#include <memory>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
The domain over which a Chebyshev polynomial is defined; returned by ChebyMap.getDomain.
*/
class ChebyDomain {
public:
    /**
    Construct a ChebyDomain

    @param[in] lbnd  Lower bounds of domain (one element per axis)
    @param[in] ubnd  Upper bounds of domain (one element per axis)
    */
    ChebyDomain(std::vector<double> const &lbnd, std::vector<double> const &ubnd) : lbnd(lbnd), ubnd(ubnd) {}
    std::vector<double> const lbnd;  ///< lower bound of domain (one element per axis)
    std::vector<double> const ubnd;  ///< upper bound of domain (one element per axis)
};

/**
A ChebyMap is a form of Mapping which performs a Chebyshev polynomial
transformation.  Each output coordinate is a linear combination of
Chebyshev polynomials of the first kind, of order zero up to a
specified maximum order, evaluated at the input coordinates. The
coefficients to be used in the linear combination are specified
separately for each output coordinate.

For a 1-dimensional ChebyMap, the forward transformation is defined
as follows:

f(x) = c0 T0(x') + c1 T1(x') + c2 T2(x') + ...

where:
   - Tn(x') is the nth Chebyshev polynomial of the first kind:
        - T0(x') = 1
        - T1(x') = x'
        - Tn+1(x') = 2.x'.Tn(x') + Tn-1(x')
   - x' is the inpux axis value, x, offset and scaled to the range
     [-1, 1] as x ranges over a specified bounding box, given when the
     ChebyMap is created. The input positions, x,  supplied to the
     forward transformation must fall within the bounding box - nans
     are generated for points outside the bounding box.

For an N-dimensional ChebyMap, the forward transformation is a
generalisation of the above form. Each output axis value is the sum
of `ncoeff` terms, where each term is the product of a single coefficient
value and N factors of the form `Tn(x'_i)`, where `x'_i` is the
normalised value of the i'th input axis value.

The forward and inverse transformations are defined independantly
by separate sets of coefficients, supplied when the ChebyMap is
created. If no coefficients are supplied to define the inverse
transformation, the polyTran method can instead be used to create an
inverse transformation. The inverse transformation so generated
will be a Chebyshev polynomial with coefficients chosen to minimise
the residuals left by a round trip (forward transformation followed
by inverse transformation).

### Attributes

All those of Mapping. In addition, the forward and inverse bounds can be retrieved using getDomain

Strictly speaking, ChebyMap it has all the attributes of PolyMap, but the only attributes PolyMap adds
to Mapping are iterative inverse parameters and those are ignored by ChebyMap because it does not (yet)
support an iterative inverse.
*/
class ChebyMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref ChebyMap with a specified forward and/or inverse transforms.

    The two sets of coefficients are independent of each other: the inverse transform
    need not undo the forward transform.

    @param[in] coeff_f  A @ref ChebyMap_CoefficientMatrices "matrix of coefficients" describing the
        forward transformation. If `coeff_f` is empty then no forward transformation is provided.
    @param[in] coeff_i  A  @ref ChebyMap_CoefficientMatrices "matrix of coefficients" describing the
        inverse transformation. If `coeff_i` is empty then no inverse transformation is provided,
        unless you specify suitable options to request an iterative inverse; see the
        @ref ChebyMap(ConstArray2D const &, int, std::vector<double> const &,
        std::string const &) "other constructor" for details.
    @param[in] lbnd_f  Lower bounds for input data; one element per input axis
    @param[in] ubnd_f  Upper bounds for input data; one element per input axis
    @param[in] lbnd_i  Lower bounds for output data; one element per output axis
    @param[in] ubnd_i  Upper bounds for output data; one element per output axis
    @param[in] options  Comma-separated list of attribute assignments.

    If a transform is not specified then the corresponding bounds are ignored (not even length-checked)
    and can be empty. For example if `coeff_f` is empty then `lbnd_f` and `ubnd_f` are ignored.

    @throws std::invalid_argument if neither transform is specified (coeff_f and coeff_i are both empty).
    @throws std::invalid_argument if the forward transform is specified (coeff_f is not empty)
        and lbnd_f or ubnd_f do not have nin elements.
    @throws std::invalid_argument if the inverse transform is specified (coeff_i is not empty)
        and lbnd_i or ubnd_i do not have nout elements.

    @anchor ChebyMap_CoefficientMatrices Coefficient Matrices
                                         --------------------

    The coefficients describing a forward transformation are specified as 2-dimensional ndarray,
    with one row per coefficient. Each row contains the following consecutive `(2 + nin)` values:
    - The first element is the coefficient value.
    - The next element is the integer index of the @ref ChebyMap output
        which uses the coefficient within its defining polynomial
        (the first output has index 1).
    - The remaining elements give the Chebyshev order to use with each corresponding input coordinate
        value, or 0 to ignore that input coordinate. Powers must not be negative and floating point
        values are rounded to the nearest integer.

    For example, suppose you want to make a @ref ChebyMap with 3 inputs and 2 outputs.
    Then each row of `coeff_f` must have 5 = 2 + nin elements.
    A row with values `(1.2, 2, 6, 3, 0)` describes a coefficient that increments output 2 as follows:

        `out2 += 1.2 * T6(in1') * T3(in2') * T0(in3')`

    and a row with values `(-1.5, 1, 0, 0, 0)` describes a coefficient that increments
    output 1 with a constant value of -1.5 (since all powers are 0):

        `out1 += -1.5 * T0(in1') * T0(in2') * T0(in3')`

    where inI' is the normalized value of input axis I.

    The final value of each output coordinate is the sum of all values specified by coefficients
    which increment that output coordinate, or 0 if there are no such coefficients.

    The coefficients describing the inverse transformation work the same way, of course,
    but each coefficient is described by `(2 + nout)` values.
    */
    explicit ChebyMap(ConstArray2D const &coeff_f, ConstArray2D const &coeff_i,
                      std::vector<double> const &lbnd_f, std::vector<double> const &ubnd_f,
                      std::vector<double> const &lbnd_i, std::vector<double> const &ubnd_i,
                      std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      _makeRawChebyMap(coeff_f, coeff_i, lbnd_f, ubnd_f, lbnd_i, ubnd_i, options))) {}

    /**
    Construct a @ref ChebyMap with only the forward transform specified.

    If the polynomial is invertible and you want an inverse can you call polyTran to fit one
    (at this time the iterative inverse offered by PolyMap is not available for ChebyMap).

    @param[in] coeff_f  A `(2 + nin) x ncoeff_f` @ref ChebyMap_CoefficientMatrices "matrix of coefficients"
        describing the forward transformation.
    @param[in] nout  Number of output coordinates.
    @param[in] lbnd_f  Lower bounds for input data; one element per input axis
    @param[in] ubnd_f  Upper bounds for input data; one element per input axis
    @param[in] options  Comma-separated list of attribute assignments.

    @throws std::invalid_argument if the forward transform is not defined (coeff_f is empty)
    @throws std::invalid_argument if lbnd_f or ubnd_f do not have nin elements
    */
    explicit ChebyMap(ConstArray2D const &coeff_f, int nout, std::vector<double> const &lbnd_f,
                      std::vector<double> const &ubnd_f, std::string const &options = "IterInverse=0")
            : Mapping(reinterpret_cast<AstMapping *>(
                      _makeRawChebyMap(coeff_f, nout, lbnd_f, ubnd_f, options))) {}

    virtual ~ChebyMap() {}

    /// Copy constructor: make a deep copy
    ChebyMap(ChebyMap const &) = default;
    ChebyMap(ChebyMap &&) = default;
    ChebyMap &operator=(ChebyMap const &) = delete;
    ChebyMap &operator=(ChebyMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<ChebyMap> copy() const { return std::static_pointer_cast<ChebyMap>(copyPolymorphic()); }

    /**
    Return the bounding box of the domain of a ChebyMap.

    Return the upper and lower limits of the box defining the domain
    of either the forward or inverse transformation of a ChebyMap. These
    are the values that were supplied when the ChebyMap was created.

    If the requested direction was fit using polyTran, and so does not have
    a user-specified domain bounding box, this method returns a box determined
    by calling MapBox on opposite direction's transformation.

    @param[in] forward  If true return the domain of the forward transform, else the inverse
    @throws std::runtime_error if the domain cannot be computed
    */
    ChebyDomain getDomain(bool forward) const;

    /**
    This function creates a new @ref ChebyMap which is a copy of this one,
    in which a specified transformation (forward or inverse)
    has been replaced by a new Chebyshev polynomial transformation. The
    coefficients of the new transformation are estimated by sampling
    the other transformation and performing a least squares polynomial
    fit in the opposite direction to the sampled positions and values.

    This method can only be used on (1-input,1-output) or (2-input, 2-output)
    @ref ChebyMap "ChebyMaps".

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
                    If both lbnd and ubnd are empty (the default) then they will be estimated.
                    The new polynomial will be evaluated over this rectangle. The length
                    should equal getNIn() or getNOut(), depending on `forward`.
    @param[in] ubnd  A vector holding the upper bounds of a rectangular region within
                    the ChebyMap's input space (if `forward` is false)
                    or output space (if `forward` is true).
                    If both lbnd and ubnd are empty (the default) then they will be estimated.
                    The new polynomial will be evaluated over this rectangle. The length
                    should equal getNIn() or getNOut(), depending on `forward`.

    @throws std::invalid_argument if the size of `lbnd` or `ubnd` does not match getNIn() (if `forward` false)
                    or getNOut() (if `forward` true).

    The variant that takes omits the `lbnd` and `ubnd` arguments
    uses the full domain of the polynomial whose inverse is being fit.

    @note

    The transformation to create is specified by the `forward` parameter.
    In what follows "X" refers to the inputs of the @ref ChebyMap, and "Y" to
    the outputs of the @ref ChebyMap. The forward transformation transforms
    input values (X) into output values (Y), and the inverse transformation
    transforms output values (Y) into input values (X). Within a @ref ChebyMap,
    each transformation is represented by an independent set of
    polynomials, P_f or P_i: Y=P_f(X) for the forward transformation and
    X=P_i(Y) for the inverse transformation.

    The `forward` parameter specifies the transformation to be replaced.
    If it is true, a new forward transformation is created
    by first finding the input values (X) using the inverse transformation
    (which must be available) at a regular grid of points (Y) covering a
    rectangular region of the @ref ChebyMap's output space. The coefficients of
    the required forward polynomial, Y=P_f(X), are chosen in order to
    minimise the sum of the squared residuals between the sampled values
    of Y and P_f(X).

    If `forward` is false (probably the most likely case),
    a new inverse transformation is created by
    first finding the output values (Y) using the forward transformation
    (which must be available) at a regular grid of points (X) covering a
    rectangular region of the @ref ChebyMap's input space. The coefficients of
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
    */
    ChebyMap polyTran(bool forward, double acc, double maxacc, int maxorder, std::vector<double> const &lbnd,
                      std::vector<double> const &ubnd) const;

    /**
    This method is the same as @ref polyTran except that the bounds are those originally provided
    when the  polynomial whose inverse is being fit was specified.
    */
    ChebyMap polyTran(bool forward, double acc, double maxacc, int maxorder) const;

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<ChebyMap, AstChebyMap>();
    }

    /// Construct a ChebyMap from an raw AST pointer
    ChebyMap(AstChebyMap *map);

private:
    /// Make a raw AstChebyMap with specified forward and inverse transforms.
    AstChebyMap *_makeRawChebyMap(ConstArray2D const &coeff_f, ConstArray2D const &coeff_i,
                                  std::vector<double> const &lbnd_f, std::vector<double> const &ubnd_f,
                                  std::vector<double> const &lbnd_i, std::vector<double> const &ubnd_i,
                                  std::string const &options = "") const;

    /// Make a raw AstChebyMap with a specified forward transform and an iterative inverse.
    AstChebyMap *_makeRawChebyMap(ConstArray2D const &coeff_f, int nout, std::vector<double> const &lbnd_f,
                                  std::vector<double> const &ubnd_f, std::string const &options = "") const;
};

}  // namespace ast

#endif
