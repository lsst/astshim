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
#include <complex>
#include <memory>
#include "lsst/cpputils/python.h"

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "astshim/KeyMap.h"
#include "astshim/Table.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
void wrapTable(lsst::utils::python::WrapperCollection &wrappers) {
  using PyTable = py::class_<Table, KeyMap>;
  wrappers.wrapType(
      PyTable(wrappers.module, "Tsble"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::string const &>(), "options"_a = "");

        cls.def("columnName", &Table::columnName, "index"_a);
        cls.def("columnType", &Table::columnType, "column"_a);
        cls.def("columnLength", &Table::columnLength, "column"_a);
        cls.def("columnNdim", &Table::columnNdim, "column"_a);
        cls.def("columnUnit", &Table::columnUnit, "column"_a);
        cls.def("columnLenC", &Table::columnLenC, "column"_a);
        cls.def("columnShape", &Table::columnShape, "column"_a);
        cls.def_property_readonly("nColumn", &Table::getNColumn);
        cls.def_property_readonly("nRow", &Table::getNRow);
      });
}

}  // namespace ast
