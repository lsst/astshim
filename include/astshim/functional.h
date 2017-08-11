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
#ifndef ASTSHIM_FUNCTIONAL_H
#define ASTSHIM_FUNCTIONAL_H

#include <memory>
#include <vector>

#include "astshim/FrameSet.h"
#include "astshim/Mapping.h"

/*
 * This header declares operations that treat ast objects (particularly Mapping
 * and its subclasses) as functions to be manipulated.
 */
namespace ast {

/**
 * Construct a FrameSet that performs two transformations in series.
 *
 * When used as a Mapping, the FrameSet shall apply `first`, followed by
 * `second`. Its inverse shall apply the inverse of `second`, followed by the
 * inverse of `first`. The concatenation is only valid if `first.getCurrent()`
 * and `second.getBase()` have the same number of axes.
 *
 * The new FrameSet shall contain all Frames and Mappings from `first`,
 * followed by all Frames and Mappings from `second`, preserving their
 * original order. The current frame of `first` shall be connected to the base
 * frame of `second` by a UnitMap. The new set's base frame shall be the base
 * frame of `first`, and its current frame shall be the current frame of `second`.
 *
 * The FrameSet shall be independent of the input arguments, so changes to the
 * original FrameSets (in particular, reassignments of their base or current
 * frames) shall not affect it.
 *
 * @param first, second the FrameSets to concatenate.
 * @return a pointer to a combined FrameSet as described above
 *
 * Example: if `first` has 3 frames and `second `has 4, then the result shall
 *          contain 7 frames, of which frames 1-3 are the same as frames 1-3 of
 *          `first`, in order, and frames 4-7 are the same as frames
 *          1-4 of `second`, in order.
 */
std::shared_ptr<FrameSet> append(FrameSet const& first, FrameSet const& second);

/**
 * Construct a radially symmetric mapping from a 1-dimensional mapping
 *
 * The transform will be symmetrical about the specified center.
 * The forward transform is as follows:
 * input -> unitNormMap -> input norm -> mapping1d -> output norm -> unitNormMap inverse -> output
 *                      -> unit vector ---------------------------->
 * where unitNormMap is UnitNormMap(center)
 *
 * The returned mapping will support forward and/or inverse transformation as `mapping1d` does.
 *
 * @param[in] center  Center of radial symmetry
 * @param[in] mapping1d  1-dimensional mapping
 * @returns a mapping that is radially symmetric about the center and has nIn = nOut = center.size()
 *
 * @throws std::invalid_argument if mapping1d has nIn or nOut != 1
 * @throws std::runtime_error if center is empty
 */
std::shared_ptr<Mapping> makeRadialMapping(std::vector<double> const& center, Mapping const& mapping1d);

}  // namespace ast

#endif  // ASTSHIM_FUNCTIONAL_H
