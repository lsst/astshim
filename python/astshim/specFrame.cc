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

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/Frame.h"
#include "astshim/SkyFrame.h"
#include "astshim/SpecFrame.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapSpecFrame(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PySpecFrame= nb::class_<SpecFrame, Frame>;
    wrappers.wrapType(PySpecFrame(wrappers.module, "SpecFrame"), [](auto &mod, auto &cls) {
        cls.def(nb::init<std::string const &>(), "options"_a = "");
        cls.def(nb::init<SpecFrame const &>());

        cls.def("copy", &SpecFrame::copy);

        cls.def("getAlignSpecOffset", &SpecFrame::getAlignSpecOffset);
        cls.def("getAlignStdOfRest", &SpecFrame::getAlignStdOfRest);
        cls.def("getRefDec", &SpecFrame::getRefDec);
        cls.def("getRefRA", &SpecFrame::getRefRA);
        cls.def("getRefPos", nb::overload_cast<SkyFrame const &>(&SpecFrame::getRefPos, nb::const_), "frame"_a);
        cls.def("getRefPos", nb::overload_cast<>(&SpecFrame::getRefPos, nb::const_));
        cls.def("getRestFreq", &SpecFrame::getRestFreq);
        cls.def("getSourceSys", &SpecFrame::getSourceSys);
        cls.def("getSourceVel", &SpecFrame::getSourceVel);
        cls.def("getSourceVRF", &SpecFrame::getSourceVRF);
        cls.def("getSpecOrigin", &SpecFrame::getSpecOrigin);
        cls.def("getStdOfRest", &SpecFrame::getStdOfRest);

        cls.def("setAlignSpecOffset", &SpecFrame::setAlignSpecOffset, "align"_a);
        cls.def("setAlignStdOfRest", &SpecFrame::setAlignStdOfRest, "stdOfRest"_a);
        cls.def("setRefDec", &SpecFrame::setRefDec, "refDec"_a);
        cls.def("setRefRA", &SpecFrame::setRefRA, "refRA"_a);
        cls.def("setRefPos", nb::overload_cast<SkyFrame const &, double, double>(&SpecFrame::setRefPos),
                "frame"_a, "lon"_a, "lat"_a);
        cls.def("setRefPos", nb::overload_cast<double, double>(&SpecFrame::setRefPos), "ra"_a, "dec"_a);
        cls.def("setRestFreq", nb::overload_cast<double>(&SpecFrame::setRestFreq), "freq"_a);
        cls.def("setRestFreq", nb::overload_cast<std::string const &>(&SpecFrame::setRestFreq), "freq"_a);
        cls.def("setSourceSys", &SpecFrame::setSourceSys, "system"_a);
        cls.def("setSourceVel", &SpecFrame::setSourceVel, "vel"_a);
        cls.def("setSourceVRF", &SpecFrame::setSourceVRF, "vrf"_a);
        cls.def("setSpecOrigin", &SpecFrame::setSpecOrigin, "origin"_a);
        cls.def("setStdOfRest", &SpecFrame::setStdOfRest, "stdOfRest"_a);
    });
}

}  // namespace ast
