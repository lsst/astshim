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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/PermMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapPermMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyPermMap = nb::class_<PermMap, Mapping>;
    wrappers.wrapType(PyPermMap (wrappers.module, "PermMap"), [](auto &mod, auto &cls) {
        cls.def(nb::init<std::vector<int> const &, std::vector<int> const &, std::vector<double> const &,
                        std::string const &>(),
                "inperm"_a, "outperm"_a, "constant"_a = std::vector<double>(), "options"_a = "");
        cls.def(nb::init<PermMap const &>());

        cls.def("copy", &PermMap::copy);
    });
}

}  // namespace ast
