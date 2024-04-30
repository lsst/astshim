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

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include "ndarray/nanobind.h"
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/PolyMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapPolyMap(lsst::cpputils::python::WrapperCollection &wrappers){
    using PyPolyMap = nb::class_<PolyMap, Mapping>;
    wrappers.wrapType(PyPolyMap (wrappers.module, "PolyMap"), [](auto &mod, auto &cls) {
        cls.def(nb::init<ConstArray2D const &, ConstArray2D const &, std::string const &>(), "coeff_f"_a,
                "coeff_i"_a, "options"_a = "IterInverse=0");
        cls.def(nb::init<ConstArray2D const &, int, std::string const &>(), "coeff_f"_a, "nout"_a,
                "options"_a = "IterInverse=0");
        cls.def(nb::init<PolyMap const &>());
        cls.def_prop_ro("iterInverse", &PolyMap::getIterInverse);
        cls.def_prop_ro("nIterInverse", &PolyMap::getNIterInverse);
        cls.def_prop_ro("tolInverse", &PolyMap::getTolInverse);
        cls.def("copy", &PolyMap::copy);
        cls.def("polyTran", &PolyMap::polyTran, "forward"_a, "acc"_a, "maxacc"_a, "maxorder"_a, "lbnd"_a,
                "ubnd"_a);
    });
}

}  // namespace ast
