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

namespace py = pybind11;
using namespace pybind11::literals;

#include "astshim/Frame.h"
#include "astshim/FrameSet.h"

namespace ast {

PYBIND11_PLUGIN(frameSet) {
    py::module::import("astshim.frame");

    py::module mod("frameSet", "Python wrapper for FrameSet");

    py::class_<FrameSet, std::shared_ptr<FrameSet>, Frame> cls(mod, "FrameSet");

    cls.def(py::init<Frame const &, std::string const &>(), "frame"_a, "options"_a="");
    cls.def(py::init<Frame const &, Mapping const &, Frame const &, std::string const &>(),
            "baseFrame"_a, "mapping"_a, "currentFrame"_a, "options"_a="");

    // def_readonly_static makes in only available in the class, not instances, so...
    cls.attr("BASE") = py::cast(AST__BASE);
    cls.attr("CURRENT") = py::cast(AST__CURRENT);

    cls.def("copy", &FrameSet::copy);
    cls.def("addAxes", &FrameSet::addAxes);
    cls.def("addFrame", &FrameSet::addFrame, "iframe"_a, "map"_a, "frame"_a);
    cls.def("addVariant", &FrameSet::addVariant, "map"_a, "name"_a);
    cls.def("getAllVariants", &FrameSet::getAllVariants);
    cls.def("getBase", &FrameSet::getBase);
    cls.def("getCurrent", &FrameSet::getCurrent);
    cls.def("getFrame", &FrameSet::getFrame, "iframe"_a);
    cls.def("getMapping", &FrameSet::getMapping, "ind1"_a=FrameSet::BASE, "ind2"_a=FrameSet::CURRENT);
    cls.def("getNframe", &FrameSet::getNframe);
    cls.def("getVariant", &FrameSet::getVariant);
    cls.def("mirrorVariants", &FrameSet::mirrorVariants, "iframe"_a);
    cls.def("remapFrame", &FrameSet::remapFrame, "iframe"_a, "map"_a);
    cls.def("removeFrame", &FrameSet::removeFrame, "iframe"_a);
    cls.def("renameVariant", &FrameSet::renameVariant, "name"_a);
    cls.def("setBase", &FrameSet::setBase, "ind"_a);
    cls.def("setCurrent", &FrameSet::setCurrent, "ind"_a);

    return mod.ptr();
}

}  // ast
