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
#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/MatrixMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapMatrixMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyMatrixMap = py::class_<MatrixMap, std::shared_ptr<MatrixMap>, Mapping>;
    wrappers.wrapType(PyMatrixMap(wrappers.module, "MatrixMap"), [](auto &mod, auto &cls) {
        cls.def(py::init<ConstArray2D const &, std::string const &>(), "matrix"_a, "options"_a = "");
        cls.def(py::init<std::vector<double> const &, std::string const &>(), "diag"_a, "options"_a = "");
        cls.def(py::init<MatrixMap const &>());
        cls.def("copy", &MatrixMap::copy);
    });
}

}  // namespace ast
