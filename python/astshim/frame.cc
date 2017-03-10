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

#include "astshim/CmpFrame.h"
#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/Mapping.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

void wrapDirectionPoint(py::module & mod) {
    py::class_<DirectionPoint> cls(mod, "DirectionPoint");

    cls.def(py::init<double, PointD>(), "direction"_a, "point"_a);

    cls.def_readwrite("direction", &DirectionPoint::direction);
    cls.def_readwrite("point", &DirectionPoint::point);
}

void wrapNReadValue(py::module & mod) {
    py::class_<NReadValue> cls(mod, "NReadValue");

    cls.def(py::init<int, double>(), "nread"_a, "value"_a);

    cls.def_readwrite("nread", &NReadValue::nread);
    cls.def_readwrite("value", &NReadValue::value);
}

void wrapResolvedPoint(py::module & mod) {
    py::class_<ResolvedPoint> cls(mod, "ResolvedPoint");

    cls.def(py::init<int>(), "naxes"_a);

    cls.def_readwrite("point", &ResolvedPoint::point);
    cls.def_readwrite("d1", &ResolvedPoint::d1);
    cls.def_readwrite("d2", &ResolvedPoint::d2);
}

void wrapFrameMapping(py::module & mod) {
    py::class_<FrameMapping> cls(mod, "FrameMapping");

    cls.def(py::init<std::shared_ptr<Frame>, std::shared_ptr<Mapping>>(), "frame"_a, "mapping"_a);

    cls.def_readwrite("frame", &FrameMapping::frame);
    cls.def_readwrite("mapping", &FrameMapping::mapping);
}

PYBIND11_PLUGIN(frame) {
    py::module mod("frame", "Python wrapper for Frame");

    py::module::import("astshim.mapping");

    wrapDirectionPoint(mod);
    wrapNReadValue(mod);
    wrapResolvedPoint(mod);
    wrapFrameMapping(mod);

    py::class_<Frame, std::shared_ptr<Frame>, Mapping> cls(mod, "Frame");

    cls.def(py::init<int, std::string const &>(), "naxes"_a, "options"_a="");

    cls.def("copy", &Frame::copy);
    cls.def("angle", &Frame::angle, "a"_a, "b"_a, "c"_a);
    cls.def("axAngle", &Frame::axAngle, "a"_a, "b"_a, "axis"_a);
    cls.def("axDistance", &Frame::axDistance, "axis"_a, "v1"_a, "v2"_a);
    cls.def("axOffset", &Frame::axOffset, "axis"_a, "v1"_a, "dist"_a);
    cls.def("convert", &Frame::convert, "to"_a, "domainlist"_a="");
    cls.def("distance", &Frame::distance, "point1"_a, "point2"_a);
    cls.def("findFrame", &Frame::findFrame, "template"_a, "domainlist"_a="");
    cls.def("format", &Frame::format, "axis"_a, "value"_a);
    cls.def("getActiveUnit", &Frame::getActiveUnit);
    cls.def("getAlignSystem", &Frame::getAlignSystem);
    cls.def("getBottom", &Frame::getBottom, "axis"_a);
    cls.def("getDigits", &Frame::getDigits);
    cls.def("getDirection", &Frame::getDirection, "axis"_a);
    cls.def("getDomain", &Frame::getDomain);
    cls.def("getDut1", &Frame::getDut1);
    cls.def("getEpoch", &Frame::getEpoch);
    cls.def("getFormat", &Frame::getFormat);
    cls.def("getInternalUnit", &Frame::getInternalUnit);
    cls.def("getLabel", &Frame::getLabel);
    cls.def("getMatchEnd", &Frame::getMatchEnd);
    cls.def("getMaxAxes", &Frame::getMaxAxes);
    cls.def("getMinAxes", &Frame::getMinAxes);
    cls.def("getNaxes", &Frame::getNaxes);
    cls.def("getNormUnit", &Frame::getNormUnit, "axis"_a);
    cls.def("getObsAlt", &Frame::getObsAlt);
    cls.def("getObsLat", &Frame::getObsLat);
    cls.def("getObsLon", &Frame::getObsLon);
    cls.def("getPermute", &Frame::getPermute);
    cls.def("getPreserveAxes", &Frame::getPreserveAxes);
    cls.def("getSymbol", &Frame::getSymbol, "axis"_a);
    cls.def("getSystem", &Frame::getSystem);
    cls.def("getTitle", &Frame::getTitle);
    cls.def("getTop", &Frame::getTop, "axis"_a);
    cls.def("getUnit", &Frame::getUnit, "axis"_a);
    cls.def("intersect", &Frame::intersect, "a1"_a, "a2"_a, "b1"_a, "b2"_a);
    cls.def("matchAxes", &Frame::matchAxes, "other"_a);
    cls.def("over", &Frame::over, "first"_a);
    cls.def("norm", &Frame::norm, "value"_a);
    cls.def("offset", &Frame::offset, "point1"_a, "point2"_a, "offset"_a);
    cls.def("offset2", &Frame::offset2, "point1"_a, "angle"_a, "offset"_a);
    cls.def("permAxes", &Frame::permAxes, "perm"_a);
    cls.def("pickAxes", &Frame::pickAxes, "axes"_a);
    cls.def("resolve", &Frame::resolve, "point1"_a, "point2"_a, "point3"_a);
    cls.def("setAlignSystem", &Frame::setAlignSystem, "system"_a);
    cls.def("setBottom", &Frame::setBottom, "axis"_a, "bottom"_a);
    cls.def("setDigits", (void (Frame::*)(int)) &Frame::setDigits, "digits"_a);
    cls.def("setDigits", (void (Frame::*)(int, int)) &Frame::setDigits, "digits"_a, "axis"_a);
    cls.def("setDirection", &Frame::setDirection, "direction"_a, "axis"_a);
    cls.def("setDomain", &Frame::setDomain, "domain"_a);
    cls.def("setDut1", &Frame::setDut1, "dut1"_a);
    cls.def("setEpoch", (void (Frame::*)(double)) &Frame::setEpoch, "epoch"_a);
    cls.def("setEpoch", (void (Frame::*)(std::string const &)) &Frame::setEpoch, "epoch"_a);
    cls.def("setFormat", &Frame::setFormat, "axis"_a, "format"_a);
    cls.def("setLabel", &Frame::setLabel, "axis"_a, "label"_a);
    cls.def("setMatchEnd", &Frame::setMatchEnd, "match"_a);
    cls.def("setMaxAxes", &Frame::setMaxAxes, "maxAxes"_a);
    cls.def("setMinAxes", &Frame::setMinAxes, "minAxes"_a);
    cls.def("setObsAlt", &Frame::setObsAlt, "altitude"_a);
    cls.def("setObsLat", &Frame::setObsLat, "latitude"_a);
    cls.def("setObsLon", &Frame::setObsLon, "longitutde"_a);
    cls.def("setActiveUnit", &Frame::setActiveUnit, "enable"_a);
    cls.def("setPermute", &Frame::setPermute, "permute"_a);
    cls.def("setPreserveAxes", &Frame::setPreserveAxes, "preserve"_a);
    cls.def("setSymbol", &Frame::setSymbol, "axis"_a, "symbol"_a);
    cls.def("setSystem", &Frame::setSystem, "system"_a);
    cls.def("setTitle", &Frame::setTitle, "title"_a);
    cls.def("setTop", &Frame::setTop, "axis"_a, "top"_a);
    cls.def("setUnit", &Frame::setUnit, "axis"_a, "unit"_a);
    cls.def("unformat", &Frame::unformat, "axis"_a, "str"_a);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
