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

#include "astshim/Mapping.h"
#include "astshim/QuadApprox.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

PYBIND11_PLUGIN(_quadApprox) {
    py::module mod("_quadApprox", "Python wrapper for QuadApprox");

    py::class_<QuadApprox> cls(mod, "QuadApprox");

    cls.def(py::init<Mapping const &, std::vector<double> const &, std::vector<double> const &, int, int>(),
            "map"_a, "lbnd"_a, "ubnd"_a, "nx"_a = 3, "ny"_a = 3);

    cls.def_readonly("fit", &QuadApprox::fit);
    cls.def_readonly("rms", &QuadApprox::rms);

    return mod.ptr();
}

}  // ast
