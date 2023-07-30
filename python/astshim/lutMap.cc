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

#include "astshim/Mapping.h"
#include "astshim/LutMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapLutMap(lsst::utils::python::WrapperCollection &wrappers) {
    using PyLutMap = py::class_<LutMap, Mapping>;
    wrappers.wrapType(PyLutMap(wrappers.module, "LutMap"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::vector<double> const &, double, double, std::string const &>(), "lut"_a, "start"_a,
                "inc"_a, "options"_a = "");
        cls.def(py::init<LutMap const &>());
        cls.def_property_readonly("lutEpsilon", &LutMap::getLutEpsilon);
        cls.def_property_readonly("lutInterp", &LutMap::getLutInterp);
        cls.def("copy", &LutMap::copy);
    });
}

}  // namespace ast
