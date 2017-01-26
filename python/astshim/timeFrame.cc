/*
 * LSST Data Management System
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 * See the COPYRIGHT file
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
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "astshim/Frame.h"
#include "astshim/TimeFrame.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

PYBIND11_PLUGIN(_timeFrame) {
    py::module mod("_timeFrame", "Python wrapper for TimeFrame");

    py::class_<TimeFrame, std::shared_ptr<TimeFrame>, Frame> cls(mod, "TimeFrame");

    cls.def(py::init<std::string const &>(), "options"_a="");

    cls.def("copy", &TimeFrame::copy);
    cls.def("currentTime", &TimeFrame::currentTime);

    cls.def("getAlignTimeScale", &TimeFrame::getAlignTimeScale);
    cls.def("getLTOffset", &TimeFrame::getLTOffset);
    cls.def("getTimeOrigin", &TimeFrame::getTimeOrigin);
    cls.def("getTimeScale", &TimeFrame::getTimeScale);

    cls.def("setAlignTimeScale", &TimeFrame::setAlignTimeScale, "scale"_a);
    cls.def("setLTOffset", &TimeFrame::setLTOffset, "offset"_a);
    cls.def("setTimeOrigin", &TimeFrame::setTimeOrigin, "origin"_a);
    cls.def("setTimeScale", &TimeFrame::setTimeScale, "scale"_a);

    return mod.ptr();
}

}  // ast
