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
#ifndef ASTSHIM_MAPSPLIT_H
#define ASTSHIM_MAPSPLIT_H

#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/detail/utils.h"

namespace ast {
class Mapping;

/**
A Mapping split off as a subset of another Mapping.
*/
class MapSplit {
public:
    /**
    Construct a MapSplit

    The subset is specified by choosing a subset of inputs from
    an existing Mapping. Such a split is only possible if the specified inputs
    correspond to some subset of the original Mapping's outputs. That is, there
    must exist a subset of the Mapping outputs for which each output
    depends only on the selected Mapping inputs, and not on any of the
    inputs which have not been selected. Also, any output which is not in
    this subset must not depend on any of the selected inputs.

    @param[in] map  Mapping to split.
    @param[in] in  Indices of inputs of `map` to pick.
        Each element should have a value in the range [1, map.getNIn()].

    @throws std::runtime_error if `map` cannot be split as specified.
    */
    explicit MapSplit(Mapping const &map, std::vector<int> const &in);

    MapSplit(MapSplit const &) = default;
    MapSplit(MapSplit &&) = default;
    MapSplit &operator=(MapSplit const &) = default;
    MapSplit &operator=(MapSplit &&) = default;

    /**
    The Mapping that was split off.
    */
    std::shared_ptr<Mapping> splitMap;
    /**
    Indices of the inputs of the original mapping were picked for the split mapping

    This is a copy of the `in` argument of the constructor.
    */
    std::vector<int> origIn;
    /**
    Indices of the outputs of the original mapping which are fed by the picked inputs.

    This will contain splitMap->getNOut() elements, each in the range [1, nout of the original mapping].
    The `i`th element holds the index within the original mapping which corresponds to
    the `i`th output of the split mapping. For example if the 1st output of the split mapping
    came from the 5th output of the original mapping, then origOut[0] = 5 (0 because
    vectors use 0-based indexing, and 5 because AST index values are 1-based).
    */
    std::vector<int> origOut;
};

}  // namespace ast

#endif
