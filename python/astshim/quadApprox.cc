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
#include <nanobind/stl/vector.h>

#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/QuadApprox.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapQuadApprox(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyQuadApprox =  nb::class_<QuadApprox>;
    wrappers.wrapType(PyQuadApprox(wrappers.module, "QuadApprox"), [](auto &mod, auto &cls) {
        cls.def(nb::init<Mapping const &, std::vector<double> const &, std::vector<double> const &, int, int>(),
                "map"_a, "lbnd"_a, "ubnd"_a, "nx"_a = 3, "ny"_a = 3);
        cls.def_ro("fit", &QuadApprox::fit);
        cls.def_ro("rms", &QuadApprox::rms);
    });
}

}  // namespace ast
