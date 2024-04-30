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
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "ndarray/nanobind.h"
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/ChebyMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapChebyMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyChebyDomain = nb::class_<ChebyDomain>;
    wrappers.wrapType(PyChebyDomain(wrappers.module, "ChebyDomain"), [](auto &mod, auto &cls) {

        cls.def(nb::init<std::vector<double> const &, std::vector<double> const &>(), "lbnd"_a, "ubnd"_a);
        cls.def_ro("lbnd", &ChebyDomain::lbnd);
        cls.def_ro("ubnd", &ChebyDomain::ubnd);
    });

    using PyChebyMap =  nb::class_<ChebyMap, Mapping>;
    wrappers.wrapType(PyChebyMap(wrappers.module, "ChebyMap"), [](auto &mod, auto &cls) {

        cls.def(nb::init<ConstArray2D const &, ConstArray2D const &, std::vector<double> const &,
                        std::vector<double> const &, std::vector<double> const &, std::vector<double> const &,
                        std::string const &>(),
                "coeff_i"_a, "coeff_i"_a, "lbnds_f"_a, "ubnds_f"_a, "lbnds_i"_a, "ubnds_i"_a, "options"_a = "");
        cls.def(nb::init<ConstArray2D const &, int, std::vector<double> const &, std::vector<double> const &,
                        std::string const &>(),
                "coeff_i"_a, "nout"_a, "lbnds_f"_a, "ubnds_f"_a, "options"_a = "");
        cls.def(nb::init<ChebyMap const &>());

        cls.def("copy", &ChebyMap::copy);
        cls.def("getDomain", &ChebyMap::getDomain, "forward"_a);
        cls.def("polyTran",
                nb::overload_cast<bool, double, double, int, std::vector<double> const &,
                        std::vector<double> const &>(&ChebyMap::polyTran, nb::const_),
                "forward"_a, "acc"_a, "maxacc"_a, "maxorder"_a, "lbnd"_a, "ubnd"_a);
        cls.def("polyTran", nb::overload_cast<bool, double, double, int>(&ChebyMap::polyTran, nb::const_),
                "forward"_a, "acc"_a, "maxacc"_a, "maxorder"_a);
    });
}

}  // namespace ast
