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

namespace py = pybind11;
using namespace pybind11::literals;

#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/FrameDict.h"
#include "astshim/Mapping.h"

namespace ast {
namespace {

PYBIND11_PLUGIN(frameDict) {
    py::module mod("frameDict", "Python wrapper for FrameDict");

    py::module::import("astshim.frameSet");

    py::class_<FrameDict, std::shared_ptr<FrameDict>, FrameSet> cls(mod, "FrameDict");

    cls.def(py::init<Frame const &, std::string const &>(), "frame"_a, "options"_a = "");
    cls.def(py::init<Frame const &, Mapping const &, Frame const &, std::string const &>(), "baseFrame"_a,
            "mapping"_a, "currentFrame"_a, "options"_a = "");
    cls.def(py::init<FrameSet const &>(), "frameSet"_a);

    cls.def("copy", &FrameDict::copy);

    cls.def("addFrame", (void (FrameDict::*)(int, Mapping const &, Frame const &)) & FrameDict::addFrame,
            "iframe"_a, "map"_a, "frame"_a);
    cls.def("addFrame",
            (void (FrameDict::*)(std::string const &, Mapping const &, Frame const &)) & FrameDict::addFrame,
            "domain"_a, "map"_a, "frame"_a);
    cls.def("getAllDomains", &FrameDict::getAllDomains);
    cls.def("getFrame", (std::shared_ptr<Frame>(FrameDict::*)(int, bool) const) & FrameDict::getFrame,
            "index"_a, "copy"_a = true);
    cls.def("getFrame",
            (std::shared_ptr<Frame>(FrameDict::*)(std::string const &, bool) const) & FrameDict::getFrame,
            "domain"_a, "copy"_a = true);
    cls.def("getMapping", (std::shared_ptr<Mapping>(FrameDict::*)(int, int) const) & FrameDict::getMapping,
            "from"_a = FrameDict::BASE, "to"_a = FrameDict::CURRENT);
    cls.def("getMapping",
            (std::shared_ptr<Mapping>(FrameDict::*)(int, std::string const &) const) & FrameDict::getMapping,
            "from"_a, "to"_a);
    cls.def("getMapping",
            (std::shared_ptr<Mapping>(FrameDict::*)(std::string const &, int) const) & FrameDict::getMapping,
            "from"_a, "to"_a);
    cls.def("getMapping",
            (std::shared_ptr<Mapping>(FrameDict::*)(std::string const &, std::string const &) const) &
                    FrameDict::getMapping,
            "from"_a, "to"_a);
    cls.def("getIndex", &FrameDict::getIndex, "domain"_a);
    cls.def("hasDomain", &FrameDict::hasDomain, "domain"_a);
    cls.def("mirrorVariants", (void (FrameDict::*)(int)) & FrameDict::mirrorVariants, "index"_a);
    cls.def("mirrorVariants", (void (FrameDict::*)(std::string const &)) & FrameDict::mirrorVariants,
            "domain"_a);
    cls.def("remapFrame", (void (FrameDict::*)(int, Mapping &)) & FrameDict::remapFrame, "index"_a, "map"_a);
    cls.def("remapFrame", (void (FrameDict::*)(std::string const &, Mapping &)) & FrameDict::remapFrame,
            "domain"_a, "map"_a);
    cls.def("removeFrame", (void (FrameDict::*)(int)) & FrameDict::removeFrame, "index"_a);
    cls.def("removeFrame", (void (FrameDict::*)(std::string const &)) & FrameDict::removeFrame, "domain"_a);
    cls.def("setBase", (void (FrameDict::*)(int)) & FrameDict::setBase, "index"_a);
    cls.def("setBase", (void (FrameDict::*)(std::string const &)) & FrameDict::setBase, "domain"_a);
    cls.def("setCurrent", (void (FrameDict::*)(int)) & FrameDict::setCurrent, "index"_a);
    cls.def("setCurrent", (void (FrameDict::*)(std::string const &)) & FrameDict::setCurrent, "domain"_a);
    cls.def("setDomain", &FrameDict::setDomain, "domain"_a);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
