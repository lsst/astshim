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
#ifndef ASTSHIM_MAPPING_H
#define ASTSHIM_MAPPING_H

#include <memory>
#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/detail/utils.h"
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
- @ref Mapping_NIn "NIn": Number of input coordinates for a Mapping
- @ref Mapping_NOut "NOut": Number of output coordinates for a Mapping
- @ref Mapping_Report "Report": Report transformed coordinates to stdout?
- @ref Mapping_TranForward "TranForward": Is the forward transformation defined?
- @ref Mapping_TranInverse "TranInverse": Is the inverse transformation defined?
*/
class Mapping : public Object {
    friend class Object;

public:
    virtual ~Mapping() {}

    /// Copy constructor: make a deep copy
    Mapping(Mapping const &) = default;
    Mapping(Mapping &&) = default;
    Mapping &operator=(Mapping const &) = delete;
    Mapping &operator=(Mapping &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<Mapping> copy() const { return std::static_pointer_cast<Mapping>(copyPolymorphic()); }

    /**
    Get @ref Mapping_NIn "NIn": the number of input axes
    */
    int getNIn() const { return getI("NIn"); }

    /**
    Get @ref Mapping_NOut "NOut": the number of output axes
    */
    int getNOut() const { return getI("NOut"); }

    /**
    Get @ref Mapping_IsSimple "IsSimple": has the mapping been simplified?
    */
    bool getIsSimple() const { return getI("IsSimple"); }

    /**
    Is this an inverted mapping?

    Note: this gets the @ref Mapping_Invert "Invert" attribute.
    This method is not called `getInvert` because that sounds like it might return the inverse.
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
    Is the forward transform available?

    @note This gets the @ref Mapping_TranForward "TranForward" attribute,
    but is named `hasForward` instead of `getTranForward` for clarity,
    since it does not return a transform.
    */
    bool hasForward() const { return getB("TranForward"); }

    /**
    Is the inverse transform available?

    @note This gets the @ref Mapping_TranInverse "TranInverse" attribute,
    but is named `hasInverse` instead of `getTranInverse` for clarity,
    since it does not return a transform.
    */
    bool hasInverse() const { return getB("TranInverse"); }

    /**
    Get an inverse mapping

    An inverse mapping is a deep copy of a mapping whose @ref Mapping_Invert "Invert" attribute
    has been toggled, as indicated by @ref isInverted. This swaps the meaning of "input" and "output",
    and of "forward" and "inverse". Thus it swaps the behavior of @ref applyForward and @ref applyInverse,
    @ref getNIn and @ref getNOut, @ref hasForward and @ref hasInverse and so on.

    Note that the inverse mapping contains exactly the same model coefficients as the original,
    but they are used by @ref applyInverse instead of @ref applyForward. Thus for example if a @ref ZoomMap
    has a zoom factor of 4.0 then its inverse also reports a zoom factor of 4.0 (despite behaving
    like an uninverted @ref ZoomMap with zoom factor of 0.25).
    */
    std::shared_ptr<Mapping> inverted() const;

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

    @throws std::runtime_error if the forward transformation cannot be modeled to within the specified `tol`.
    */
    Array2D linearApprox(PointD const &lbnd, PointD const &ubnd, double tol) const;

    /**
    Return a series compound mapping this(first(input)) containing shallow copies of the original

    @param[in] next  the mapping whose input is the output of this mapping

    @throws std::invalid_argument if the number of input axes of `next` does not match
        the number of output axes of this mapping.

    @warning The contained mappings are shallow copies (just like AST);
    if you want deep copies then make them manually.
    */
    SeriesMap then(Mapping const &next) const;

    /**
    Return a parallel compound mapping containing shallow copies of the original

    The resulting mapping has `getNIn() + next.getNIn()` inputs and `getNOut() + next.getNOut()` outputs.
    The first `getNIn()` axes of input are transformed by this mapping, producing the first `getNOut()` axes
    of output. The remaining axes of input are processed by `next`, resulting in the remaining axes of output.

    The name comes the way vectors are sometimes shown for matrix multiplication:
    vertically, with the first axis at the bottom and the last axis at the top.

    @param[in] next  the mapping that processes the final `next.getNin()` axes of input to produce
    the final `next.getNout()` axes of output.

    @warning The contained mappings are shallow copies (just like AST);
    if you want deep copies then make them manually.
    */
    ParallelMap under(Mapping const &next) const;

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
                    or `nan` if the value cannot be calculated.
    */
    double rate(PointD const &at, int ax1, int ax2) const {
        detail::assertEqual(at.size(), "at.size", static_cast<std::size_t>(getNIn()), "nIn");
        double result = astRate(getRawPtr(), const_cast<double *>(at.data()), ax1, ax2);
        assertOK();
        return result;
    }

    /**
    Set @ref Mapping_Report "Report": report transformed coordinates to stdout?
    */
    void setReport(bool report) { setB("Report", report); }

    /**
    Return a simplied version of the mapping (which may be a compound Mapping such as a CmpMap).

    Simplification eliminates redundant computational steps and merges separate steps which can be performed
    more efficiently in a single operation. As a simple example, a Mapping which multiplied coordinates by 5,
    and then multiplied the result by 10, could be simplified to a single step which multiplied by 50.
    Similarly, a Mapping which multiplied by 5, and then divided by 5, could be reduced to
    a simple copying operation.

    This function should typically be applied to Mappings which have undergone substantial processing
    or have been formed by merging other Mappings. It is of potential benefit, for example,
    in reducing execution time if applied before using a Mapping to transform a large number of coordinates.

    @note
    * If the supplied Mapping is a FrameSet, the returned Mapping will be a deep copy of the supplied
      FrameSet in which all the inter-Frame Mappings have been simplified.
    * Mappings that have a set value for their ref Object_Ident "Ident" attribute
      are unchanged by simplification. This is so that their individual identity is preserved.
      This restriction does not apply to the simplification of Frames.
    * The returned mapping is always independent of the original (a deep copy), unlike astSimplify.
    */
    std::shared_ptr<Mapping> simplified() const {
        AstObject *rawSimpMap = reinterpret_cast<AstObject *>(astSimplify(getRawPtr()));
        assertOK(rawSimpMap);
        return Object::fromAstObject<Mapping>(rawSimpMap, true);
    }

    /**
    Perform a forward transformation on 2-D array, putting the results into a pre-allocated 2-D array

    @param[in] from  input coordinates, with dimensions (nPts, nIn)
    @param[out] to  transformed coordinates, with dimensions (nPts, nOut)
    */
    void applyForward(ConstArray2D const &from, Array2D const &to) const { _tran(from, true, to); }

    /**
    Perform a forward transformation on a 2-D array, returning the results as a new array

    @param[in] from  input coordinates, with dimensions (nPts, nIn)
    @return the results as a new array with dimensions (nPts, nOut)
    */
    Array2D applyForward(ConstArray2D const &from) const {
        Array2D to = ndarray::allocate(getNOut(), from.getSize<1>());
        _tran(from, true, to);
        return to;
    }

    /**
    Perform a forward transformation on a vector, returning the results as a new vector

    @param[in] from  input coordinates as a vector, with axes adjacent, e.g. x0, y0, x1, y1...xn, yn
    @return the results as a new vector
    */
    std::vector<double> applyForward(std::vector<double> const &from) const {
        auto fromArr = arrayFromVector(from, getNIn());
        std::vector<double> to(fromArr.getSize<1>() * getNOut());
        auto toArr = arrayFromVector(to, getNOut());
        _tran(fromArr, true, toArr);
        return to;
    }

    /**
    Perform an inverse transformation on a 2-D array, putting the results into a pre-allocated 2-D array

    @param[in] from  input coordinates, with dimensions (nPts, nOut)
    @param[out] to  transformed coordinates, with dimensions (nPts, nIn)
    */
    void applyInverse(ConstArray2D const &from, Array2D const &to) const { _tran(from, false, to); }

    /**
    Perform an inverse transformation on a 2-D array, returning the results as a new 2-D array

    @param[in] from  output coordinates, with dimensions (nPts, nOut)
    @return the results as a new array with dimensions (nPts, nIn)
    */
    Array2D applyInverse(ConstArray2D const &from) const {
        Array2D to = ndarray::allocate(getNIn(), from.getSize<1>());
        _tran(from, false, to);
        return to;
    }

    /**
    Perform an inverse transformation on a vector, returning the results as a new vector

    @param[in] from  input coordinates as a vector, with axes adjacent, e.g. x0, y0, x1, y1...xn, yn
    @return the results as a new vector
    */
    std::vector<double> applyInverse(std::vector<double> const &from) const {
        auto fromArr = arrayFromVector(from, getNOut());
        std::vector<double> to(fromArr.getSize<1>() * getNIn());
        auto toArr = arrayFromVector(to, getNIn());
        _tran(fromArr, false, toArr);
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
    @param[in] to  Computed points, with dimensions (nPts, nOut), where nPts the desired number of points
    */
    void tranGridForward(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix,
                         Array2D const &to) const {
        _tranGrid(lbnd, ubnd, tol, maxpix, true, to);
    }

    /**
    Transform a grid of points in the inverse direction, returning the results as a new Array2D

    See the overload of tranGridForward that outputs the data as the last argument
    for more information
    */
    Array2D tranGridForward(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix, int nPts) const {
        Array2D to = ndarray::allocate(nPts, getNOut());
        _tranGrid(lbnd, ubnd, tol, maxpix, true, to);
        return to;
    }

    /**
    Transform a grid of points in the inverse direction

    See tranGridForward for the arguments, swapping nIn and nOut
    */
    void tranGridInverse(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix,
                         Array2D const &to) const {
        _tranGrid(lbnd, ubnd, tol, maxpix, false, to);
    }

    /**
    Transform a grid of points in the inverse direction

    See tranGridForward for the arguments, swapping nIn and nOut
    */
    Array2D tranGridInverse(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix, int nPts) const {
        Array2D to = ndarray::allocate(nPts, getNIn());
        _tranGrid(lbnd, ubnd, tol, maxpix, false, to);
        return to;
    }

protected:
    /**
    Construct a mapping from a pointer to a raw AST subclass of AstMapping
    */
    explicit Mapping(AstMapping *rawMap) : Object(reinterpret_cast<AstObject *>(rawMap)) {
        assertOK();
        if (!astIsAMapping(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a Mapping";
            throw std::invalid_argument(os.str());
        }
    }

    // Protected implementation of deep-copy.
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return std::static_pointer_cast<Mapping>(copyImpl<Mapping, AstMapping>());
    }

    /**
    Return a deep copy of one of the two component mappings.

    This is intended to be exposed by classes that need it (e.g. @ref CmpMap, @ref CmpFrame and @ref TranMap)
    as `operator[]`.

    @tparam Class  astshim class of returned object, typically Mapping or Frame.
    @param[in] i  Index: 0 for the first mapping, 1 for the second
    @param[in] copy  If true make a deep copy, else a shallow copy

    @throws std::invalid_argument if `i` is not 0 or 1.
    @throws std::runtime_error if this mapping is not a compound mapping.
    */
    template <typename Class>
    std::shared_ptr<Class> decompose(int i, bool copy) const;

private:
    /**
    Implement applyForward and applyInverse, putting the results into a pre-allocated 2-D array.

    @param[in] from  input coordinates, with dimensions (nPts, nIn)
    @param[in] doForward  if true then perform a forward transform, else inverse
    @param[out] to  transformed coordinates, must be pre-allocated with dimensions (nPts, nOut)
    */
    void _tran(ConstArray2D const &from, bool doForward, Array2D const &to) const;

    /**
    Implementat tranGridForward and tranGridInverse, which see.
    */
    void _tranGrid(PointI const &lbnd, PointI const &ubnd, double tol, int maxpix, bool doForward,
                   Array2D const &to) const;
};

}  // namespace ast

#endif
