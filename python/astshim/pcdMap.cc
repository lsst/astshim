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
#include "astshim/PcdMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapPcdMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyPcdMap =  py::class_<PcdMap, std::shared_ptr<PcdMap>, Mapping>;
    wrappers.wrapType(PyPcdMap(wrappers.module, "PcdMap"), [](auto &mod, auto &cls) {
        cls.def(py::init<double, std::vector<double> const &, std::string const &>(), "disco"_a, "pcdcen"_a,
                "options"_a = "");
        cls.def(py::init<PcdMap const &>());
        cls.def_property_readonly("disco", &PcdMap::getDisco);
        cls.def_property_readonly("pcdCen", py::overload_cast<>(&PcdMap::getPcdCen, py::const_));
        cls.def("copy", &PcdMap::copy);
    });
}

}  // namespace ast
