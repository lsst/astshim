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

namespace nb = nanobind;
using namespace nanobind::literals;

#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "lsst/cpputils/python.h"

namespace ast {

void wrapFrameSet(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyFrameSet = nb::class_<FrameSet, Frame>;
    wrappers.wrapType(PyFrameSet(wrappers.module, "FrameSet"), [](auto &mod, auto &cls) {
        cls.def(nb::init<Frame const &, std::string const &>(), "frame"_a, "options"_a = "");
        cls.def(nb::init<Frame const &, Mapping const &, Frame const &, std::string const &>(), "baseFrame"_a,
                "mapping"_a, "currentFrame"_a, "options"_a = "");
        cls.def(nb::init<Frame const &>());

        // def_prop_ro_static makes in only available in the class, not instances, so...
        cls.attr("BASE") = nb::cast(AST__BASE);
        cls.attr("CURRENT") = nb::cast(AST__CURRENT);
        cls.attr("NOFRAME") = nb::cast(AST__NOFRAME);

        cls.def_prop_rw("base", &FrameSet::getBase, &FrameSet::setBase);
        cls.def_prop_rw("current", &FrameSet::getCurrent, &FrameSet::setCurrent);
        cls.def_prop_ro("nFrame", &FrameSet::getNFrame);

        cls.def("copy", &FrameSet::copy);
        cls.def("addAxes", &FrameSet::addAxes);
        cls.def("addFrame", &FrameSet::addFrame, "iframe"_a, "map"_a, "frame"_a);
        cls.def("addVariant", &FrameSet::addVariant, "map"_a, "name"_a);
        cls.def("getAllVariants", &FrameSet::getAllVariants);
        cls.def("getFrame", &FrameSet::getFrame, "iframe"_a, "copy"_a = true);
        cls.def("getMapping", &FrameSet::getMapping, "from"_a = FrameSet::BASE, "to"_a = FrameSet::CURRENT);
        cls.def("getVariant", &FrameSet::getVariant);
        cls.def("mirrorVariants", &FrameSet::mirrorVariants, "iframe"_a);
        cls.def("remapFrame", &FrameSet::remapFrame, "iframe"_a, "map"_a);
        cls.def("removeFrame", &FrameSet::removeFrame, "iframe"_a);
        cls.def("renameVariant", &FrameSet::renameVariant, "name"_a);
    });
}

}  // namespace ast
