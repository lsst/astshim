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
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "astshim/Mapping.h"
#include "astshim/PolyMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(polyMap) {
    py::module mod("polyMap", "Python wrapper for PolyMap");

    py::module::import("astshim.mapping");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<PolyMap, std::shared_ptr<PolyMap>, Mapping> cls(mod, "PolyMap");

    cls.def(py::init<ndarray::Array<double, 2, 2> const &, ndarray::Array<double, 2, 2> const &,
                     std::string const &>(),
            "coeff_f"_a, "coeff_i"_a, "options"_a = "IterInverse=0");
    cls.def(py::init<ndarray::Array<double, 2, 2> const &, int, std::string const &>(), "coeff_f"_a, "nout"_a,
            "options"_a = "IterInverse=0");

    cls.def("copy", &PolyMap::copy);
    cls.def("getIterInverse", &PolyMap::getIterInverse);
    cls.def("getNiterInverse", &PolyMap::getNiterInverse);
    cls.def("getTolInverse", &PolyMap::getTolInverse);
    cls.def("polyTran", &PolyMap::polyTran);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
