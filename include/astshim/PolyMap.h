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
#ifndef ASTSHIM_POLYMAP_H
#define ASTSHIM_POLYMAP_H

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
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
    Construct a @ref PolyMap with specified forward and inverse transforms.

    @param[in] coeff_f  A `(2 + nin) x ncoeff_f` matrix of coefficients.
            Each row of `2 + nin` elements describe a single coefficient of the forward transformation.
            Within each such row, the first element is the coefficient value; the next element is
            the integer index of the @ref PolyMap output which uses the coefficient within its defining polynomial
            (the first output has index 1); the remaining elements of
            the row give the integer powers to use with each input coordinate value (powers
            must not be negative, and floating point values are rounded to the nearest integer).

            For instance, if the @ref PolyMap has 3 inputs and 2 outputs, each row consisting
            of 5 elements, A row such as "(1.2, 2.0, 1.0, 3.0, 0.0)" describes a coefficient
            with value 1.2 which is used within the definition of output 2.  The output value
            is incremented by the product of the coefficient value, the value of input coordinate
            1 raised to the power 1, and the value of input coordinate 2 raised to the power
            3. Input coordinate 3 is not used since its power is specified as 0.  As another
            example, the row "(-1.0, 1.0, 0.0, 0.0, 0.0)" adds a constant value -1.0 to output 1
            (it is a constant value since the power for every input axis is given as 0).

            Each final output coordinate value is the sum of the terms described
            by the `ncoeff_f` columns in the supplied array.
    @param[in] coeff_i  A (2 + nout) x ncoeff_i` matrix of coefficients.
            Each row of `2 + nout` adjacent elements describe a single coefficient of
            the inverse transformation, using the same schame as `coeff_f`,
            except that "inputs" and "outputs" are transposed.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit PolyMap(
        ndarray::Array<double, 2, 2> const & coeff_f,
        ndarray::Array<double, 2, 2> const & coeff_i,
        std::string const & options=""
    ) :
        Mapping(reinterpret_cast<AstMapping *>(_makeRawPolyMap(coeff_f, coeff_i, options)))
    {}

    /**
    Construct a @ref PolyMap with only the forward transform specified.
    The inverse may be determined by an iterative approximation.

    @param[in] coeff_f  A `(2 + nin) x ncoeff_f` matrix of coefficients.
            Each row of `2 + nin` elements describe a single coefficient of the forward transformation.
            Within each such row:
            - The first element is the coefficient value.
            - The next element is the integer index of the @ref PolyMap output
                which uses the coefficient within its defining polynomial
                (the first output has index 1).
            - The remaining elements give the integer power to use with each corresponding input coordinate
                value, or 0 to ignore that input coordinate. Powers must not be negative and floating point
                values are rounded to the nearest integer.

            For instance, if the @ref PolyMap has 3 inputs and 2 outputs, each row consisting
            of 5 elements, a row such as "(1.2, 2.0, 1.0, 3.0, 0.0)" describes a coefficient
            with value 1.2 which is used within the definition of output 2.  The output value
            is incremented by the product of the coefficient value, the value of input coordinate
            1 raised to the power 1, and the value of input coordinate 2 raised to the power
            3. Input coordinate 3 is not used since its power is specified as 0.  As another
            example, the row "(-1.0, 1.0, 0.0, 0.0, 0.0)" adds a constant value -1.0 to output 1
            (it is a constant value since the power for every input axis is given as 0).

            Each final output coordinate value is the sum of the terms described
            by the `ncoeff_f` columns in the supplied array.
    @param[in] nout  Number of output coordinates.
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit PolyMap(
        ndarray::Array<double, 2, 2> const & coeff_f,
        int nout,
        std::string const & options=""
    ) :
        Mapping(reinterpret_cast<AstMapping *>(_makeRawPolyMap(coeff_f, nout, options)))
    {}

    virtual ~PolyMap() {}

    PolyMap(PolyMap const &) = delete;
    PolyMap(PolyMap &&) = default;
    PolyMap & operator=(PolyMap const &) = delete;
    PolyMap & operator=(PolyMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<PolyMap> copy() const {
        return std::static_pointer_cast<PolyMap>(_copyPolymorphic());
    }

    /// Get @ref PolyMap_IterInverse "IterInverse": provide an iterative inverse transformation?
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
    PolyMap polyTran(
        bool forward,
        double acc,
        double maxacc,
        int maxorder,
        std::vector<double> const & lbnd,
        std::vector<double> const & ubnd
    ) const {
        int desSize = forward ? getNin() : getNout();
        if (lbnd.size() != desSize) {
            std::ostringstream os;
            os << "lbnd.size() = " << lbnd.size() << " != " << desSize
                << " = " << (forward ? "getNin()" : "getNout()");
            throw std::invalid_argument(os.str());
        }
        if (ubnd.size() != desSize) {
            std::ostringstream os;
            os << "ubnd.size() = " << ubnd.size() << " != " << desSize
                << " = " << (forward ? "getNin()" : "getNout()");
            throw std::invalid_argument(os.str());
        }

        void *map = astPolyTran(this->getRawPtr(), static_cast<int>(forward), acc, maxacc, maxorder,
                                lbnd.data(), ubnd.data());
        return PolyMap(reinterpret_cast<AstPolyMap *>(map));
    }

protected:
    virtual std::shared_ptr<Object> _copyPolymorphic() const override {
        return _copyImpl<PolyMap, AstPolyMap>();
    }    

    /// Construct a PolyMap from an raw AST pointer
    PolyMap(AstPolyMap * map) :
        Mapping(reinterpret_cast<AstMapping *>(map))
    {
        if (!astIsAPolyMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClass() << ", which is not a PolyMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    /// Make a raw AstPolyMap with forward and inverse transforms.
    AstPolyMap * _makeRawPolyMap(
        ndarray::Array<double, 2, 2>  const & coeff_f,
        ndarray::Array<double, 2, 2>  const & coeff_i,
        std::string const & options=""
    ) {
        const int nin = coeff_f.getSize<1>() - 2;
        const int ncoeff_f = coeff_f.getSize<0>();
        const int nout = coeff_i.getSize<1>() - 2;
        const int ncoeff_i = coeff_i.getSize<0>();

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

        return astPolyMap(nin, nout, ncoeff_f, coeff_f.getData(),
                          ncoeff_i, coeff_i.getData(), options.c_str());
    }

    /// Make a raw AstPolyMap with only a forward transform.
    AstPolyMap * _makeRawPolyMap(
        ndarray::Array<double, 2, 2>  const & coeff_f,
        int nout,
        std::string const & options=""
    ) {
        const int nin = coeff_f.getSize<1>() - 2;
        const int ncoeff_f = coeff_f.getSize<0>();
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
};

}  // namespace ast

#endif
