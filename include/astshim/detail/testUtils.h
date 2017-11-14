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
#ifndef ASTSHIM_DETAIL_TESTUTILS_H
#define ASTSHIM_DETAIL_TESTUTILS_H

#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "astshim/base.h"
#include "astshim/FrameDict.h"
#include "astshim/FrameSet.h"

namespace ast {
namespace detail {

/**
Make a FrameDict from a copy of a FrameSet

This exists purely to test FrameDict(FrameSet const &) from Python,
as the standard pybind11 wrapper isn't sufficient to exercise a bug that was found.
*/
FrameDict makeFrameDict(FrameSet const & frameSet) {
    return FrameDict(frameSet);
}

}  // namespace detail
}  // namespace ast

#endif