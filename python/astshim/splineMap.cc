/*
 * This file is part of astshim.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/SplineMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
void wrapSplineMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PySplineMap=py::classh<SplineMap, Mapping>;
    wrappers.wrapType(PySplineMap(wrappers.module, "SplineMap"), [](auto &mod, auto &cls) {

        cls.def(py::init<int, int, int, int, std::vector<double> const &, std::vector<double> const &,
                       std::vector<double> const &, std::vector<double> const &, std::string const &>(), "kx"_a, 
                       "ky"_a, "nx"_a, "ny"_a, "tx"_a, "ty"_a, "cu"_a, "cv"_a, "options"_a = "");
        cls.def(py::init<SplineMap const &>());
        cls.def("copy", &SplineMap::copy);
        cls.def_property_readonly("invNIter", &SplineMap::getInvNiter);
        cls.def_property_readonly("outUnit", &SplineMap::getOutUnit);
        cls.def_property_readonly("invTol", &SplineMap::getInvTol);
    });
}

}  // namespace ast
