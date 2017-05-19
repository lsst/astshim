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
#ifndef ASTSHIM_MAPBOX_H
#define ASTSHIM_MAPBOX_H

#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/detail/utils.h"

namespace ast {
class Mapping;

/**
Object to compute the bounding box which just encloses another box
after it has been transformed by a mapping.

@warning  The points in `xl` and `xu` are not predictable if more than one input value
    gives the same output boundary value.
*/
class MapBox {
public:
    /**
    Find a bounding box for a Mapping.

    Find the "bounding box" which just encloses another box
    after it has been transformed by a mapping.
    A typical use might be to calculate the size which an image would have
    after being transformed by this mapping.

    @param[in] map  Mapping for which to find the output bounding box.
    @param[in] lbnd  Lower bound of the input box.
    @param[in] ubnd  Upper bound of the input box.
       Note that it is permissible for the lower bound to exceed the
       corresponding upper bound, as the values will simply be swapped before use.
    @param[in] minOutCoord  Minimum output coordinate axis for which to compute
        an output bounding box, starting from 1
    @param[in] maxOutCoord  Maximum output coordinate axis for which to compute
        an output bounding box, starting from 1,
        or 0 for all remaining output coordinate axes (in which case
        the field of the same name will be set to the number of outputs)

    @return A @ref MapBox containing the computed outputs and a copy of the inputs.

    @throws std::invalid_argument if minOutCoord is not in the range [1, getNOut()]
        or maxOutCoord is neither 0 nor in the range [minOutCoord, getNOut()].
    @throws std::runtime_error if the required output bounds cannot be
        found. Typically, this might occur if all the input points which
        the function considers turn out to be invalid (see above). The
        number of points considered before generating such an error is
        quite large, however, so this is unlikely to occur by accident
        unless valid points are restricted to a very small subset of the
        input coordinate space.

    ### Notes

    - Any input points which are transformed by the Mapping to give
    output coordinates containing the value `AST__BAD` are regarded as
    invalid and are ignored, They will make no contribution to
    determining the output bounds, even although the nominated
    output coordinate might still have a valid value at such points.
    */
    explicit MapBox(Mapping const &map, std::vector<double> const &lbnd, std::vector<double> const &ubnd,
                    int minOutCoord = 1, int maxOutCoord = 0);

    MapBox(MapBox const &) = default;
    MapBox(MapBox &&) = default;
    MapBox &operator=(MapBox const &) = default;
    MapBox &operator=(MapBox &&) = default;

    std::vector<double> lbndIn;  ///< Lower bound of the input box.
    std::vector<double> ubndIn;  ///< Upper bound of the input box.
    /// Minimum output coordinate axis for which to compute an output bounding box, starting from 1
    int minOutCoord;
    /// Maximum output coordinate axis for which to compute an output bounding box, starting from 1
    int maxOutCoord;
    std::vector<double> lbndOut;  ///< Lower bound of the output box.
    std::vector<double> ubndOut;  ///< Upper bound of the output box.
    Array2D xl;  ///< 2-d array of [out coord, an input point at which the lower bound occurred]
    Array2D xu;  ///< 2-d array of [out coord, an input point at which the upper bound occurred]

private:
    /// Compute the outputs
    void _compute(Mapping const &map, std::vector<double> const &lbnd, std::vector<double> const &ubnd,
                  int minOutCoord = 1, int maxOutCoord = 0);
};

}  // namespace ast

#endif
