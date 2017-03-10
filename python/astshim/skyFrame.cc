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
#include "astshim/Mapping.h"
#include "astshim/SkyFrame.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(skyFrame) {
    py::module mod("skyFrame", "Python wrapper for SkyFrame");

    py::module::import("astshim.frame");

    py::class_<SkyFrame, std::shared_ptr<SkyFrame>, Frame> cls(mod, "SkyFrame");

    cls.def(py::init<std::string const &>(), "options"_a="");

    cls.def("copy", &SkyFrame::copy);

    cls.def("getAlignOffset", &SkyFrame::getAlignOffset);
    cls.def("getAsTime", &SkyFrame::getAsTime, "axis"_a);
    cls.def("getEquinox", &SkyFrame::getEquinox);
    cls.def("getIsLatAxis", &SkyFrame::getIsLatAxis, "axis"_a);
    cls.def("getIsLonAxis", &SkyFrame::getIsLonAxis, "axis"_a);
    cls.def("getLatAxis", &SkyFrame::getLatAxis);
    cls.def("getLonAxis", &SkyFrame::getLonAxis);
    cls.def("getNegLon", &SkyFrame::getNegLon);
    cls.def("getProjection", &SkyFrame::getProjection);
    cls.def("getSkyRef", (double (SkyFrame::*)(int) const) &SkyFrame::getSkyRef, "axis"_a);
    cls.def("getSkyRef", (std::vector<double> (SkyFrame::*)() const) &SkyFrame::getSkyRef);
    cls.def("getSkyRefIs", &SkyFrame::getSkyRefIs);
    cls.def("getSkyRefP", (double (SkyFrame::*)(int) const) &SkyFrame::getSkyRefP, "axis"_a);
    cls.def("getSkyRefP", (std::vector<double> (SkyFrame::*)() const) &SkyFrame::getSkyRefP);
    cls.def("getSkyTol", &SkyFrame::getSkyTol);

    cls.def("setAlignOffset", &SkyFrame::setAlignOffset);
    cls.def("setAsTime", &SkyFrame::setAsTime, "axis"_a, "asTime"_a);
    cls.def("setEquinox", &SkyFrame::setEquinox);
    cls.def("setNegLon", &SkyFrame::setNegLon);
    cls.def("setProjection", &SkyFrame::setProjection);
    cls.def("setSkyRef", &SkyFrame::setSkyRef);
    cls.def("setSkyRefIs", &SkyFrame::setSkyRefIs);
    cls.def("setSkyRefP", &SkyFrame::setSkyRefP);
    cls.def("setSkyTol", &SkyFrame::setSkyTol);

    cls.def("skyOffsetMap", &SkyFrame::skyOffsetMap);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
