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
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/Frame.h"
#include "astshim/SkyFrame.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapSkyFrame(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PySkyFrame = nb::class_<SkyFrame, Frame>;
    wrappers.wrapType(PySkyFrame(wrappers.module, "SkyFrame"), [](auto &mod, auto &cls) {

        cls.def(nb::init<std::string const &>(), "options"_a = "");
        cls.def(nb::init<SkyFrame const &>());

        cls.def("copy", &SkyFrame::copy);

        cls.def_prop_rw("alignOffset", &SkyFrame::getAlignOffset, &SkyFrame::setAlignOffset);
        cls.def_prop_rw("asTime", [](SkyFrame const &self) {
                             return std::make_pair(self.getAsTime(1), self.getAsTime(2));
                         },
                         [](SkyFrame &self, std::pair<bool, bool> asTime) {
                             self.setAsTime(1, asTime.first);
                             self.setAsTime(2, asTime.second);
                         });
        cls.def_prop_rw("alignOffset", &SkyFrame::getAlignOffset, &SkyFrame::setAlignOffset);
        cls.def_prop_rw("equinox", &SkyFrame::getEquinox, &SkyFrame::setEquinox);
        cls.def_prop_ro("latAxis", &SkyFrame::getLatAxis);
        cls.def_prop_ro("lonAxis", &SkyFrame::getLonAxis);
        cls.def_prop_rw("negLon", &SkyFrame::getNegLon, &SkyFrame::setNegLon);
        cls.def_prop_rw("projection", &SkyFrame::getProjection, &SkyFrame::setProjection);
        cls.def_prop_rw("skyRefIs", &SkyFrame::getSkyRefIs, &SkyFrame::setSkyRefIs);
        cls.def_prop_rw("skyTol", &SkyFrame::getSkyTol, &SkyFrame::setSkyTol);

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
