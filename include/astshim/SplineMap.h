/*
 * This file is part of astshim.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef ASTSHIM_SPLINEMAP_H
#define ASTSHIM_SPLINEMAP_H

#include <memory>
#include <vector>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
SplineMap is a @ref Mapping which performs a 2-in, 2-out Mapping defined by a pair of two-
dimensional B-spline surfaces. The inverse transformation is fit iteratively, and requires that
the output axes are a small (i.e. much less than the range of the input values) perturbation of the
input axes. 
### Attributes

All those of @ref Mapping plus:

- "InvNiter": Maximum number of iterations for iterative inverse.
- "InvTol": Target relative error for iterative inverse.
- "OutUnit": Determines how out-of-bounds input positions are handled. If 
true, input points outside the bounding box are copied unchanged to output. If false, the output
for these points will be nan.
*/
class SplineMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a @ref SplineMap with the forward transform specified by the supplied b-splines and
    the inverse transform fit iteratively.

    @param[in] kx  Polynomial order in X input direction.
    @param[in] ky  Polynomial order in Y input direction.
    @param[in] nx  Number of coefficients in X input direction.
    @param[in] ny  Number of coefficients in Y input direction
    @param[in] tx  Array of knot positions for X input.
    @param[in] ty  Array of knot positions for Y input.
    @param[in] cu  Array of coefficients defining U output.
    @param[in] cv  Array of coefficients defining V output.
    @param[in] options Comma-separated list of attribute assignments.

    @throws std::invalid_argument if the polynomial order of number of coefficients in either 
        direction are negative.
    @throws std::invalid_argument if the polynomial order plus the number of coefficients does
        not equal the number of knot positions for a given direction.
    @throws std::invalid_argument if the product of the number of coefficients in each direction
        does not equal the number of coefficients provided.

    The coefficients cu and cv are 1-dimensional arrays.
    */
    explicit SplineMap(int kx, int ky, int nx, int ny, std::vector<double> const &tx, std::vector<double> const &ty,
                       std::vector<double> const &cu, std::vector<double> const &cv, std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(
                      _makeRawSplineMap(kx, ky, nx, ny, tx, ty, cu, cv, options))) {}

    ~SplineMap() override = default;

    /// Copy constructor: make a deep copy
    SplineMap(SplineMap const &) = default;
    SplineMap(SplineMap &&) = default;
    SplineMap &operator=(SplineMap const &) = delete;
    SplineMap &operator=(SplineMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<SplineMap> copy() const { return std::static_pointer_cast<SplineMap>(copyPolymorphic()); }

    /// Get "InvNiter": Get maximum number of iterations for iterative inverse.
    int getInvNiter() const { return getI("InvNiter"); }

    /// Get "OutUnit": Is the out-of-bounds position transform an identity map?
    int getOutUnit() const { return getI("OutUnit"); }

    /// Get "InvTol": Get target relative error for iterative inverse.
    double getInvTol() const { return getD("InvTol"); }


protected:
    std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<SplineMap, AstSplineMap>();
    }

    /// Construct a SplineMap from an raw AST pointer
    explicit SplineMap(AstSplineMap *map);

private:
    /// Make a raw AstSplineMap with a specified forward transform and an iterative inverse.
    AstSplineMap *_makeRawSplineMap(int kx, int ky, int nx, int ny, std::vector<double> const &tx, std::vector<double> const &ty,
                       std::vector<double> const &cu, std::vector<double> const &cv, std::string const &options = "") const;
};

}  // namespace ast

#endif
