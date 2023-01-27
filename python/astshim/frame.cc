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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lsst/cpputils/python.h"

#include "astshim/CmpFrame.h"
#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/Mapping.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapDirectionPoint(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<DirectionPoint>(wrappers.module, "DirectionPoint"), [](auto &mod, auto &cls) {
        cls.def(py::init<double, PointD>(), "direction"_a, "point"_a);
        cls.def_readwrite("direction", &DirectionPoint::direction);
        cls.def_readwrite("point", &DirectionPoint::point);
    });
}

void wrapNReadValue(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<NReadValue>(wrappers.module, "NReadValue"), [](auto &mod, auto &cls) {
        cls.def(py::init<int, double>(), "nread"_a, "value"_a);
        cls.def_readwrite("nread", &NReadValue::nread);
        cls.def_readwrite("value", &NReadValue::value);
    });
}

void wrapResolvedPoint(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<ResolvedPoint> (wrappers.module, "ResolvedPoint"), [](auto &mod, auto &cls) {
        cls.def(py::init<int>(), "naxes"_a);
        cls.def_readwrite("point", &ResolvedPoint::point);
        cls.def_readwrite("d1", &ResolvedPoint::d1);
        cls.def_readwrite("d2", &ResolvedPoint::d2);
    });
}

void wrapFrameMapping(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<FrameMapping>(wrappers.module, "FrameMapping"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::shared_ptr<Frame>, std::shared_ptr<Mapping>>(), "frame"_a, "mapping"_a);
        cls.def_readwrite("frame", &FrameMapping::frame);
        cls.def_readwrite("mapping", &FrameMapping::mapping);
    });
}

void wrapFrame(lsst::utils::python::WrapperCollection &wrappers) {
    wrapDirectionPoint(wrappers);
    wrapNReadValue(wrappers);
    wrapResolvedPoint(wrappers);
    wrapFrameMapping(wrappers);

    using PyFrame = py::class_<Frame, std::shared_ptr<Frame>, Mapping>;
    wrappers.wrapType(PyFrame(wrappers.module, "Frame"), [](auto &mod, auto &cls) {
        cls.def(py::init<int, std::string const &>(), "naxes"_a, "options"_a = "");
        cls.def(py::init<Frame const &>());

        cls.def_property("activeUnit", &Frame::getActiveUnit, &Frame::setActiveUnit);
        cls.def_property("alignSystem", &Frame::getAlignSystem, &Frame::setAlignSystem);
        cls.def_property("domain", &Frame::getDomain, &Frame::setDomain);
        cls.def_property("dut1", &Frame::getDut1, &Frame::setDut1);
        cls.def_property("epoch", &Frame::getEpoch, py::overload_cast<double>(&Frame::setEpoch));
        cls.def_property("matchEnd", &Frame::getMatchEnd, &Frame::setMatchEnd);
        cls.def_property("maxAxes", &Frame::getMaxAxes, &Frame::setMaxAxes);
        cls.def_property("minAxes", &Frame::getMinAxes, &Frame::setMinAxes);
        cls.def_property_readonly("nAxes", &Frame::getNAxes);
        cls.def_property("obsAlt", &Frame::getObsAlt, &Frame::setObsAlt);
        cls.def_property("obsLat", &Frame::getObsLat, &Frame::setObsLat);
        cls.def_property("obsLon", &Frame::getObsLon, &Frame::setObsLon);
        cls.def_property("permute", &Frame::getPermute, &Frame::setPermute);
        cls.def_property("preserveAxes", &Frame::getPreserveAxes, &Frame::setPreserveAxes);
        cls.def_property("system", &Frame::getSystem, &Frame::setSystem);
        cls.def_property("title", &Frame::getTitle, &Frame::setTitle);

        cls.def("copy", &Frame::copy);
        cls.def("angle", &Frame::angle, "a"_a, "b"_a, "c"_a);
        cls.def("axAngle", &Frame::axAngle, "a"_a, "b"_a, "axis"_a);
        cls.def("axDistance", &Frame::axDistance, "axis"_a, "v1"_a, "v2"_a);
        cls.def("axOffset", &Frame::axOffset, "axis"_a, "v1"_a, "dist"_a);
        cls.def("convert", &Frame::convert, "to"_a, "domainlist"_a = "");
        cls.def("distance", &Frame::distance, "point1"_a, "point2"_a);
        cls.def("findFrame", &Frame::findFrame, "template"_a, "domainlist"_a = "");
        cls.def("format", &Frame::format, "axis"_a, "value"_a);
        cls.def("getBottom", &Frame::getBottom, "axis"_a);
        cls.def("getDigits", py::overload_cast<>(&Frame::getDigits, py::const_));
        cls.def("getDigits", py::overload_cast<int>(&Frame::getDigits, py::const_), "axis"_a);
        cls.def("getDirection", &Frame::getDirection, "axis"_a);
        cls.def("getFormat", &Frame::getFormat, "axis"_a);
        cls.def("getInternalUnit", &Frame::getInternalUnit);
        cls.def("getLabel", &Frame::getLabel);
        cls.def("getSymbol", &Frame::getSymbol, "axis"_a);
        cls.def("getNormUnit", &Frame::getNormUnit, "axis"_a);
        cls.def("getSymbol", &Frame::getSymbol, "axis"_a);
        cls.def("getTop", &Frame::getTop, "axis"_a);
        cls.def("getUnit", &Frame::getUnit, "axis"_a);
        cls.def("intersect", &Frame::intersect, "a1"_a, "a2"_a, "b1"_a, "b2"_a);
        cls.def("matchAxes", &Frame::matchAxes, "other"_a);
        cls.def("under", &Frame::under, "next"_a);
        cls.def("norm", &Frame::norm, "value"_a);
        cls.def("offset", &Frame::offset, "point1"_a, "point2"_a, "offset"_a);
        cls.def("offset2", &Frame::offset2, "point1"_a, "angle"_a, "offset"_a);
        cls.def("permAxes", &Frame::permAxes, "perm"_a);
        cls.def("pickAxes", &Frame::pickAxes, "axes"_a);
        cls.def("resolve", &Frame::resolve, "point1"_a, "point2"_a, "point3"_a);
        cls.def("setDigits", py::overload_cast<int>(&Frame::setDigits), "digits"_a);
        cls.def("setDigits", py::overload_cast<int, int>(&Frame::setDigits), "axis"_a, "digits"_a);
        cls.def("setDirection", &Frame::setDirection, "direction"_a, "axis"_a);

        // keep setEpoch(string); use the epoch property to deal with it as a float
        cls.def("setEpoch", py::overload_cast<std::string const &>(&Frame::setEpoch), "epoch"_a);
        cls.def("setFormat", &Frame::setFormat, "axis"_a, "format"_a"format");
        cls.def("setLabel", &Frame::setLabel, "axis"_a, "label"_a);
        cls.def("setSymbol", &Frame::setSymbol, "axis"_a, "symbol"_a);
        cls.def("setTop", &Frame::setTop, "axis"_a, "top"_a);
        cls.def("setUnit", &Frame::setUnit, "axis"_a, "unit"_a);
        cls.def("unformat", &Frame::unformat, "axis"_a, "str"_a);
    });
}

}  // namespace ast
