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
#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"

#include "astshim/MapBox.h"
#include "astshim/Mapping.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapMapBox(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyMapBox = py::class_<MapBox>;
    wrappers.wrapType(PyMapBox(wrappers.module, "MapBox"), [](auto &mod, auto &cls) {
        cls.def(py::init<Mapping const &, std::vector<double> const &, std::vector<double> const &, int, int>(),
                "map"_a, "lbnd"_a, "ubnd"_a, "minOutCoord"_a = 1, "maxOutCoord"_a = 0);
        cls.def_readonly("lbndIn", &MapBox::lbndIn);
        cls.def_readonly("ubndIn", &MapBox::ubndIn);
        cls.def_readonly("minOutCoord", &MapBox::minOutCoord);
        cls.def_readonly("maxOutCoord", &MapBox::maxOutCoord);
        cls.def_readonly("lbndOut", &MapBox::lbndOut);
        cls.def_readonly("ubndOut", &MapBox::ubndOut);
        cls.def_readonly("xl", &MapBox::xl);
        cls.def_readonly("xu", &MapBox::xu);
    });
}

}  // namespace ast
