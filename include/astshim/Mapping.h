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
#ifndef ASTSHIM_MAPPING_H
#define ASTSHIM_MAPPING_H

#include <memory>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/detail.h"
#include "astshim/Object.h"

namespace ast {

class ParallelMap;
class SeriesMap;

/**
An abstract base class for objects which transform one set of coordinates to another.

Mapping is used to describe the relationship which exists between two different
coordinate systems and to implement operations which make use of
this (such as transforming coordinates and resampling grids of data).

### Attributes

In addition to those attributes common to @ref Object,
Mapping also has the following attributes:
- @ref Mapping_Invert "Invert": Is the mapping inverted?
- @ref Mapping_IsLinear "IsLinear": Is the Mapping linear?
- @ref Mapping_IsSimple "IsSimple": Has the Mapping been simplified?
- @ref Mapping_Nin "Nin": Number of input coordinates for a Mapping
- @ref Mapping_Nout "Nout": Number of output coordinates for a Mapping
- @ref Mapping_Report "Report": Report transformed coordinates to stdout?
- @ref Mapping_TranForward "TranForward": Is the forward transformation defined?
- @ref Mapping_TranInverse "TranInverse": Is the inverse transformation defined?
*/
class Mapping : public Object {
public:
    /**
    Construct a mapping from a pointer to a raw AST subclass of AstMapping

    This constructor is intended for use by subclasses (but cannot be made protected
    without listing the subclasses as friend classes).
    */
    explicit Mapping(AstMapping * mapping) :
        Object(reinterpret_cast<AstObject *>(mapping))
    {
        assertOK();
        if (!astIsAMapping(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClass() << ", which is not a Mapping";
            throw std::invalid_argument(os.str());
        }
    }

    /// Cast an object to a Mapping if possible, else throw std::runtime_error
    explicit Mapping(Object & obj) : Mapping(detail::shallowCopy<AstMapping>(obj.getRawPtr())) {}

    virtual ~Mapping() {}

    Mapping(Mapping const &) = default;
    Mapping(Mapping &&) = default;
    Mapping & operator=(Mapping const &) = default;
    Mapping & operator=(Mapping &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<Mapping> copy() const { return _copy<Mapping, AstMapping>(); }

    /**
    Get @ref Mapping_Nin "Nin": the number of input axes
    */
    int getNin() const { return getI("Nin"); }

    /**
    Get @ref Mapping_Nout "Nout": the number of output axes
    */
    int getNout() const { return getI("Nout"); }

    /**
    Get @ref Mapping_IsSimple "IsSimple": has the mapping been simplified?
    */
    bool getIsSimple() const { return getI("IsSimple"); }

    /**
    Is this an inverted mapping?

    Note: this gets the @ref Mapping_Invert "Invert" attribute.
    This method is not called `getInvert` because that is too similar to @ref getInverse
    and because astshim attempts to discourage use of the attribute.
    */
    bool isInverted() const { return getB("Invert"); }

    /**
    Get @ref Mapping_IsLinear "IsLinear": is the Mapping linear?
    */
    bool getIsLinear() const { return getB("IsLinear"); }

    /**
    Get @ref Mapping_Report "Report": report transformed coordinates to stdout?
    */
    bool getReport() const { return getB("Report"); }

    /**
    Get @ref Mapping_TranForward "TranForward": is the forward transform defined?
    */
    bool getTranForward() const { return getB("TranForward"); }

    /**
    Get @ref Mapping_TranInverse "TranInverse": is the inverse transform defined?
    */
    bool getTranInverse() const { return getB("TranInverse"); }

    /// Get an inverse mapping
    std::shared_ptr<Mapping> getInverse() const {
        auto rawCopy = reinterpret_cast<AstMapping *>(astCopy(getRawPtr()));
        astInvert(rawCopy);
        return std::make_shared<Mapping>(rawCopy);
    }

    /**
    Compute a linear approximation to the forward transformation.

    @param[in] lbnd  Input point defining the lower bounds of the box over which the
                        linear approximation is computed.
    @param[in] ubnd  Input point defining the upper bounds of the box over which the
                        linear approximation is computed.
    @param[in] tol  The maximum permitted deviation from linearity, expressed as apositive Cartesian
                    displacement in the output coordinate space. If a linear fit to the forward
                    transformation of the Mapping deviates from the true transformation by more than
                    this amount at any point which is tested, then raise an exception.

    @return The co-efficients of the linear approximation to the specified transformation,
            as an 1 + nIn x nOut array.
            The first index is [constant, gradiant for input 1, gradiant for input 2...]
            and the second index is [output 1, output 2, ...].
            For example, if the Mapping has 2 inputs and 3 outputs then the coefficients are:

                X_out = fit[0, 0] + fit[1, 0] X_in + fit[2, 0] Y_in
                Y_out = fit[0, 1] + fit[1, 1] X_in + fit[2, 1] Y_in
                Z_out = fit[0, 2] + fit[1, 2] X_in + fit[2, 2] Y_in

    @throw std::runtime_error if the forward transformation cannot be modeled to within the specified `tol`.
    */
    Array2D linearApprox(
        PointD const & lbnd,
        PointD const & ubnd,
        double tol
    ) const {
        int const nIn = getNin();
        int const nOut = getNout();
        detail::assertEqual(lbnd.size(), "lbnd.size", nIn, "nIn");
        detail::assertEqual(ubnd.size(), "ubnd.size", nIn, "nIn");
        Array2D fit = ndarray::allocate(ndarray::makeVector(1 + nIn, nOut));
        int isOK = astLinearApprox(getRawPtr(), lbnd.data(), ubnd.data(), tol, fit.getData());
        if (!isOK) {
            throw std::runtime_error("Mapping not sufficiently linear");
        }
        assertOK();
        return fit;
    }


    /**
    Return a series compound mapping this(first(input)).

    @param[in] first  the mapping whose output is the input of this mapping

    @throw std::invalid_argument if the number of output axes of `first` does not match
        the number of input axes of this mapping.
    */
    SeriesMap of(Mapping const & first) const;

    /**
    Return a parallel compound mapping

    The resulting mapping has first.getNin() + this.getNin() inputs and next.getNout() + this.getNout()
    outputs. The first getNin() axes of input are transformed by the `first` mapping, producing the
    first getNout() axes of the output. The remaining axes of input are processed by this mapping,
    resulting in the remaining axes of output.

    @param[in] first  the mapping that processes axes 1 through `first.getNin()`
    */
    ParallelMap over(Mapping const & first) const;

    /**
    Evaluate the rate of change of the Mapping with respect to a specified input, at a specified position.

    The result is estimated by interpolating the function using a fourth order polynomial
    in the neighbourhood of the specified position. The size of the neighbourhood used
    is chosen to minimise the RMS residual per unit length between the interpolating polynomial
    and the supplied Mapping function. This method produces good accuracy but can involve evaluating
    the Mapping 100 or more times.

    @param[in] at  The input position at which the rate of change is to be evaluated.
    @param[in] ax1  The index of the output for which the rate of change is to be found
                    (1 for first output).
    @param[in] ax2  The index of the input which is to be varied in order to find the rate of change
                    (1 for the first input).

    @return The rate of change of Mapping output `ax1` with respect to input `ax2`, evaluated at `at`,
                    or AST__BAD if the value cannot be calculated.
    */
    double rate(
        PointD const & at,
        int ax1,
        int ax2
    ) const {
        detail::assertEqual(at.size(), "at.size", getNin(), "nIn");
        double result = astRate(getRawPtr(), const_cast<double *>(at.data()), ax1, ax2);
        assertOK();
        return result;
    }

    /**
    Set @ref Mapping_Report "Report": report transformed coordinates to stdout?
    */
    void setReport(bool report) { setB("Report", report); }

    /**
    Simplify the mapping (which may be a compound Mapping such as a CmpMap)

    Simplfy eliminates redundant computational steps and merge separate steps which can be performed
    more efficiently in a single operation. As a simple example, a Mapping which multiplied coordinates by 5,
    and then multiplied the result by 10, could be simplified to a single step which multiplied by 50.
    Similarly, a Mapping which multiplied by 5, and then divided by 5, could be reduced to
    a simple copying operation.

    This function should typically be applied to Mappings which have undergone substantial processing
    or have been formed by merging other Mappings. It is of potential benefit, for example,
    in reducing execution time if applied before using a Mapping to transform a large number of coordinates.
    */
    Mapping simplify() const {
        void * simpPtr = astSimplify(getRawPtr());
        Mapping simp(reinterpret_cast<AstMapping *>(simpPtr));
        assertOK();
        return simp;
    }

    /**
    Perform a forward transformation, putting the results into a pre-allocated array

    @param[in] from  input coordinates, with dimensions (nPts, nIn)
    @param[in] to  transformed coordinates, with dimensions (nPts, nOut)
    */
    void tran(
        Array2D const & from,
        Array2D & to
    ) const {
        _tran(from, true, to);
    }

    /**
    Perform a forward transformation, returning the results as a new array

    @param[in] from  input coordinates, with dimensions (nPts, nIn)
    @return the results as a new array with dimensions (nPts, nOut)
    */
    Array2D tran(
        Array2D const & from
    ) const {
        Array2D to = ndarray::allocate(from.getSize<0>(), getNout());
        _tran(from, true, to);
        return to;
    }

    /**
    Perform an inverse transformation

    @param[in] from  input coordinates, with dimensions (nPts, nOut)
    @param[in] to  transformed coordinates, with dimensions (nPts, nIn)
    */
    void tranInverse(
        Array2D const & from,
        Array2D & to
    ) const {
        _tran(from, false, to);
    }

    /**
    Perform an inverse transformation, returning the results as a new array

    @param[in] from  output coordinates, with dimensions (nPts, nOut)
    @return the results as a new array with dimensions (nPts, nIn)
    */
    Array2D tranInverse(
        Array2D const & from
    ) const {
        Array2D to = ndarray::allocate(from.getSize<0>(), getNin());
        _tran(from, false, to);
        return to;
    }

    /**
    Transform a grid of points in the forward direction

    @param[in] lbnd  The coordinates of the centre of the first pixel in the input grid along each dimension,
                size = nIn
    @param[in] ubnd  The coordinates of the centre of the last pixel in the input grid along each dimension,
                size = nIn
    @param[in] tol  The maximum tolerable geometrical distortion which may be introduced as a result of
                approximating non-linear Mappings by a set of piece-wise linear transformations.
                This should be expressed as a displacement within the output coordinate system of the Mapping.

                If piece-wise linear approximation is not required, a value of zero may be given.
                This will ensure that the Mapping is used without any approximation, but may increase
                execution time.

                If the value is too high, discontinuities between the linear approximations used
                in adjacent panel will be higher.  If this is a problem, reduce the tolerance
                value used.
    @param[in] maxpix  A value which specifies an initial scale size (in input grid points) for the adaptive
                algorithm which approximates non-linear Mappings with piece-wise linear transformations.
                Normally, this should be a large value (larger than any dimension of the region
                of the input grid being used).  In this case, a first attempt to approximate the
                Mapping by a linear transformation will be made over the entire input region.
                If a smaller value is used, the input region will first be divided into sub-regions
                whose size does not exceed " maxpix" grid points in any dimension.  Only at this
                point will attempts at approximation commence.
                This value may occasionally be useful in preventing false convergence of the adaptive
                algorithm in cases where the Mapping appears approximately linear on large scales,
                but has irregularities (e.g.  holes) on smaller scales.  A value of, say, 50 to
                100 grid points can also be employed as a safeguard in general-purpose software,
                since the effect on performance is minimal.
                If too small a value is given, it will have the effect of inhibiting linear approximation
                altogether (equivalent to setting " tol" to zero).  Although this may degrade
                performance, accurate results will still be obtained.
    @param[in] to  Computed points, with dimensions (nPts, nOut)
    */
    void tranGridForward(
        PointI const & lbnd,
        PointI const & ubnd,
        double tol,
        int maxpix,
        Array2D & to
    ) const {
        _tranGrid(lbnd, ubnd, tol, maxpix, true, to);
    }

    /**
    Transform a grid of points in the inverse direction

    See tranGridForward for the arguments, swapping nIn and nOut
    */
    void tranGridInverse(
        PointI const & lbnd,
        PointI const & ubnd,
        double tol,
        int maxpix,
        Array2D & to
    ) const {
        _tranGrid(lbnd, ubnd, tol, maxpix, false, to);
    }

private:
    void _tran(
        Array2D const & from,
        bool doForward,
        Array2D & to
    ) const;

    void _tranGrid(
        PointI const & lbnd,
        PointI const & ubnd,
        double tol,
        int maxpix,
        bool doForward,
        Array2D & to
    ) const;
};

}  // namespace ast

#endif
