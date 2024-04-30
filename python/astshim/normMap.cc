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

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include "lsst/cpputils/python.h"

#include "astshim/Frame.h"
#include "astshim/Mapping.h"
#include "astshim/NormMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapNormMap(lsst::utils::python::WrapperCollection &wrappers) {
    using PyNormMap =  nb::class_<NormMap, Mapping>;
    wrappers.wrapType(PyNormMap(wrappers.module, "NormMap"), [](auto &mod, auto &cls) {
        cls.def(nb::init<Frame const &, std::string const &>(), "frame"_a, "options"_a = "");
        cls.def(nb::init<NormMap const &>());
        cls.def("copy", &NormMap::copy);
    });
}

}  // namespace ast
