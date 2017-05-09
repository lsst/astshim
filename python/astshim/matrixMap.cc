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
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "astshim/Mapping.h"
#include "astshim/MatrixMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(matrixMap) {
    py::module mod("matrixMap", "Python wrapper for MatrixMap");

    py::module::import("astshim.mapping");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<MatrixMap, std::shared_ptr<MatrixMap>, Mapping> cls(mod, "MatrixMap");

    cls.def(py::init<ndarray::Array<double, 2, 2> const &, std::string const &>(), "matrix"_a,
            "options"_a = "");
    cls.def(py::init<std::vector<double> const &, std::string const &>(), "diag"_a, "options"_a = "");

    cls.def("copy", &MatrixMap::copy);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
