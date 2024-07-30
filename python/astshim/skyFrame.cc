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
#include "lsst/cpputils/python.h"

#include "astshim/Frame.h"
#include "astshim/SkyFrame.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapSkyFrame(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PySkyFrame = py::class_<SkyFrame, std::shared_ptr<SkyFrame>, Frame>;
    wrappers.wrapType(PySkyFrame(wrappers.module, "SkyFrame"), [](auto &mod, auto &cls) {

        cls.def(py::init<std::string const &>(), "options"_a = "");
        cls.def(py::init<SkyFrame const &>());

        cls.def("copy", &SkyFrame::copy);

        cls.def_property("alignOffset", &SkyFrame::getAlignOffset, &SkyFrame::setAlignOffset);
        cls.def_property("asTime", [](SkyFrame const &self) {
                             return std::make_pair(self.getAsTime(1), self.getAsTime(2));
                         },
                         [](SkyFrame &self, std::pair<bool, bool> asTime) {
                             self.setAsTime(1, asTime.first);
                             self.setAsTime(2, asTime.second);
                         });
        cls.def_property("alignOffset", &SkyFrame::getAlignOffset, &SkyFrame::setAlignOffset);
        cls.def_property("equinox", &SkyFrame::getEquinox, &SkyFrame::setEquinox);
        cls.def_property_readonly("latAxis", &SkyFrame::getLatAxis);
        cls.def_property_readonly("lonAxis", &SkyFrame::getLonAxis);
        cls.def_property("negLon", &SkyFrame::getNegLon, &SkyFrame::setNegLon);
        cls.def_property("projection", &SkyFrame::getProjection, &SkyFrame::setProjection);
        cls.def_property("skyRefIs", &SkyFrame::getSkyRefIs, &SkyFrame::setSkyRefIs);
        cls.def_property("skyTol", &SkyFrame::getSkyTol, &SkyFrame::setSkyTol);

        cls.def("getAsTime", &SkyFrame::getAsTime, "axis"_a);
        cls.def("getIsLatAxis", &SkyFrame::getIsLatAxis, "axis"_a);
        cls.def("getIsLonAxis", &SkyFrame::getIsLonAxis, "axis"_a);
        cls.def("getSkyRef", &SkyFrame::getSkyRef);
        cls.def("getSkyRefP", &SkyFrame::getSkyRefP);
        cls.def("setAsTime", &SkyFrame::setAsTime, "axis"_a, "asTime"_a);
        cls.def("setEquinox", &SkyFrame::setEquinox);
        cls.def("setNegLon", &SkyFrame::setNegLon);
        cls.def("setProjection", &SkyFrame::setProjection);
        cls.def("setSkyRef", &SkyFrame::setSkyRef);
        cls.def("setSkyRefP", &SkyFrame::setSkyRefP);
        cls.def("skyOffsetMap", &SkyFrame::skyOffsetMap);
    });
}

}  // namespace ast
