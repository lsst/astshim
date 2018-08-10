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
#include <stdexcept>
#include <string>

#include "astshim/functional.h"
#include "astshim/UnitMap.h"
#include "astshim/UnitNormMap.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"

namespace ast {

std::shared_ptr<FrameSet> append(FrameSet const& first, FrameSet const& second) {
    std::shared_ptr<FrameSet> const merged = first.copy();
    std::shared_ptr<FrameSet> const newFrames = second.copy();

    newFrames->setCurrent(FrameSet::BASE);
    int const joinNAxes = first.getFrame(FrameSet::CURRENT)->getNAxes();
    merged->addFrame(FrameSet::CURRENT, UnitMap(joinNAxes), *newFrames);

    // All frame numbers from `second` have been offset in `merged` by number of frames in `first`
    int const mergedCurrent = first.getNFrame() + second.getCurrent();
    merged->setCurrent(mergedCurrent);

    return merged;
}

std::shared_ptr<Mapping> makeRadialMapping(std::vector<double> const& center, Mapping const& mapping1d) {
    auto naxes = center.size();
    if (mapping1d.getNIn() != 1) {
        throw std::invalid_argument("mapping1d has " + std::to_string(mapping1d.getNIn()) +
                                    " inputs, instead of 1");
    }
    if (mapping1d.getNOut() != 1) {
        throw std::invalid_argument("mapping1d has " + std::to_string(mapping1d.getNOut()) +
                                    " outputs, instead of 1");
    }
    auto unitNormMap = UnitNormMap(center);
    return std::make_shared<Mapping>(
            unitNormMap.then(UnitMap(naxes).under(mapping1d)).then(*unitNormMap.inverted()));
}

}  // namespace ast
