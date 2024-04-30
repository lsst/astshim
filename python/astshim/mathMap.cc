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
#include <string>
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/MathMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapMathMap(lsst::cpputils::python::WrapperCollection &wrappers) {
using PyMathMap = nb::class_<MathMap, Mapping>;
    wrappers.wrapType(PyMathMap(wrappers.module, "MathMap"), [](auto &mod, auto &cls) {
        cls.def(nb::init<int, int, std::vector<std::string> const &, std::vector<std::string> const &,
                        std::string const &>(),
                "nin"_a, "nout"_a, "fwd"_a, "ref"_a, "options"_a = "");
        cls.def(nb::init<MathMap const &>());
        cls.def_prop_ro("seed", &MathMap::getSeed);
        cls.def_prop_ro("simpFI", &MathMap::getSimpFI);
        cls.def_prop_ro("simpIF", &MathMap::getSimpIF);
        cls.def("copy", &MathMap::copy);
    });
}

}  // namespace ast
