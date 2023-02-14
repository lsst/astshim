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
#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/ChebyMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapChebyMap(lsst::utils::python::WrapperCollection &wrappers) {
    using PyChebyDomain = py::class_<ChebyDomain, std::shared_ptr<ChebyDomain>>;
    wrappers.wrapType(PyChebyDomain(wrappers.module, "ChebyDomain"), [](auto &mod, auto &cls) {

        cls.def(py::init<std::vector<double> const &, std::vector<double> const &>(), "lbnd"_a, "ubnd"_a);
        cls.def_readonly("lbnd", &ChebyDomain::lbnd);
        cls.def_readonly("ubnd", &ChebyDomain::ubnd);
    });

    using PyChebyMap =  py::class_<ChebyMap, std::shared_ptr<ChebyMap>, Mapping>;
    wrappers.wrapType(PyChebyMap(wrappers.module, "ChebyMap"), [](auto &mod, auto &cls) {

        cls.def(py::init<ConstArray2D const &, ConstArray2D const &, std::vector<double> const &,
                        std::vector<double> const &, std::vector<double> const &, std::vector<double> const &,
                        std::string const &>(),
                "coeff_i"_a, "coeff_i"_a, "lbnds_f"_a, "ubnds_f"_a, "lbnds_i"_a, "ubnds_i"_a, "options"_a = "");
        cls.def(py::init<ConstArray2D const &, int, std::vector<double> const &, std::vector<double> const &,
                        std::string const &>(),
                "coeff_i"_a, "nout"_a, "lbnds_f"_a, "ubnds_f"_a, "options"_a = "");
        cls.def(py::init<ChebyMap const &>());

        cls.def("copy", &ChebyMap::copy);
        cls.def("getDomain", &ChebyMap::getDomain, "forward"_a);
        cls.def("polyTran",
                py::overload_cast<bool, double, double, int, std::vector<double> const &,
                        std::vector<double> const &>(&ChebyMap::polyTran, py::const_),
                "forward"_a, "acc"_a, "maxacc"_a, "maxorder"_a, "lbnd"_a, "ubnd"_a);
        cls.def("polyTran", py::overload_cast<bool, double, double, int>(&ChebyMap::polyTran, py::const_),
                "forward"_a, "acc"_a, "maxacc"_a, "maxorder"_a);
    });
}

}  // namespace ast
