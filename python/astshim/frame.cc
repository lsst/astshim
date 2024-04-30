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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/CmpFrame.h"
#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/Mapping.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapDirectionPoint(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(nb::class_<DirectionPoint>(wrappers.module, "DirectionPoint"), [](auto &mod, auto &cls) {
        cls.def(nb::init<double, PointD>(), "direction"_a, "point"_a);
        cls.def_rw("direction", &DirectionPoint::direction);
        cls.def_rw("point", &DirectionPoint::point);
    });
}

void wrapNReadValue(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(nb::class_<NReadValue>(wrappers.module, "NReadValue"), [](auto &mod, auto &cls) {
        cls.def(nb::init<int, double>(), "nread"_a, "value"_a);
        cls.def_rw("nread", &NReadValue::nread);
        cls.def_rw("value", &NReadValue::value);
    });
}

void wrapResolvedPoint(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(nb::class_<ResolvedPoint> (wrappers.module, "ResolvedPoint"), [](auto &mod, auto &cls) {
        cls.def(nb::init<int>(), "naxes"_a);
        cls.def_rw("point", &ResolvedPoint::point);
        cls.def_rw("d1", &ResolvedPoint::d1);
        cls.def_rw("d2", &ResolvedPoint::d2);
    });
}

void wrapFrameMapping(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(nb::class_<FrameMapping>(wrappers.module, "FrameMapping"), [](auto &mod, auto &cls) {
        cls.def(nb::init<std::shared_ptr<Frame>, std::shared_ptr<Mapping>>(), "frame"_a, "mapping"_a);
        cls.def_rw("frame", &FrameMapping::frame);
        cls.def_rw("mapping", &FrameMapping::mapping);
    });
}

void wrapFrame(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrapDirectionPoint(wrappers);
    wrapNReadValue(wrappers);
    wrapResolvedPoint(wrappers);
    wrapFrameMapping(wrappers);

    using PyFrame = nb::class_<Frame, Mapping>;
    wrappers.wrapType(PyFrame(wrappers.module, "Frame"), [](auto &mod, auto &cls) {
        cls.def(nb::init<int, std::string const &>(), "naxes"_a, "options"_a = "");
        cls.def(nb::init<Frame const &>());

        cls.def_prop_rw("activeUnit", &Frame::getActiveUnit, &Frame::setActiveUnit);
        cls.def_prop_rw("alignSystem", &Frame::getAlignSystem, &Frame::setAlignSystem);
        cls.def_prop_rw("domain", &Frame::getDomain, &Frame::setDomain);
        cls.def_prop_rw("dut1", &Frame::getDut1, &Frame::setDut1);
        cls.def_prop_rw("epoch", &Frame::getEpoch, nb::overload_cast<double>(&Frame::setEpoch));
        cls.def_prop_rw("matchEnd", &Frame::getMatchEnd, &Frame::setMatchEnd);
        cls.def_prop_rw("maxAxes", &Frame::getMaxAxes, &Frame::setMaxAxes);
        cls.def_prop_rw("minAxes", &Frame::getMinAxes, &Frame::setMinAxes);
        cls.def_prop_ro("nAxes", &Frame::getNAxes);
        cls.def_prop_rw("obsAlt", &Frame::getObsAlt, &Frame::setObsAlt);
        cls.def_prop_rw("obsLat", &Frame::getObsLat, &Frame::setObsLat);
        cls.def_prop_rw("obsLon", &Frame::getObsLon, &Frame::setObsLon);
        cls.def_prop_rw("permute", &Frame::getPermute, &Frame::setPermute);
        cls.def_prop_rw("preserveAxes", &Frame::getPreserveAxes, &Frame::setPreserveAxes);
        cls.def_prop_rw("system", &Frame::getSystem, &Frame::setSystem);
        cls.def_prop_rw("title", &Frame::getTitle, &Frame::setTitle);

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
        cls.def("getDigits", nb::overload_cast<>(&Frame::getDigits, nb::const_));
        cls.def("getDigits", nb::overload_cast<int>(&Frame::getDigits, nb::const_), "axis"_a);
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
        cls.def("setDigits", nb::overload_cast<int>(&Frame::setDigits), "digits"_a);
        cls.def("setDigits", nb::overload_cast<int, int>(&Frame::setDigits), "axis"_a, "digits"_a);
        cls.def("setDirection", &Frame::setDirection, "direction"_a, "axis"_a);

        // keep setEpoch(string); use the epoch property to deal with it as a float
        cls.def("setEpoch", nb::overload_cast<std::string const &>(&Frame::setEpoch), "epoch"_a);
        cls.def("setFormat", &Frame::setFormat, "axis"_a, "format"_a"format");
        cls.def("setLabel", &Frame::setLabel, "axis"_a, "label"_a);
        cls.def("setSymbol", &Frame::setSymbol, "axis"_a, "symbol"_a);
        cls.def("setTop", &Frame::setTop, "axis"_a, "top"_a);
        cls.def("setUnit", &Frame::setUnit, "axis"_a, "unit"_a);
        cls.def("unformat", &Frame::unformat, "axis"_a, "str"_a);
    });
}

}  // namespace ast
