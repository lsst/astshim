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

#include "astshim/Mapping.h"
#include "astshim/LutMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(lutMap) {
    py::module mod("lutMap", "Python wrapper for LutMap");

    py::module::import("astshim.mapping");

    py::class_<LutMap, std::shared_ptr<LutMap>, Mapping> cls(mod, "LutMap");

    cls.def(py::init<std::vector<double> const &, double, double, std::string const &>(), "lut"_a, "start"_a,
            "inc"_a, "options"_a="");

    cls.def("copy", &LutMap::copy);
    cls.def("getLutEpsilon", &LutMap::getLutEpsilon);
    cls.def("getLutInterp", &LutMap::getLutInterp);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
