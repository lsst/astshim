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
#include "lsst/cpputils/python.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>


#include "astshim/KeyMap.h"
#include "astshim/Table.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {
void wrapTable(lsst::cpputils::python::WrapperCollection &wrappers) {
  using PyTable = nb::class_<Table, KeyMap>;
  wrappers.wrapType(
      PyTable(wrappers.module, "Tsble"), [](auto &mod, auto &cls) {
        cls.def(nb::init<std::string const &>(), "options"_a = "");

        cls.def("columnName", &Table::columnName, "index"_a);
        cls.def("columnType", &Table::columnType, "column"_a);
        cls.def("columnLength", &Table::columnLength, "column"_a);
        cls.def("columnNdim", &Table::columnNdim, "column"_a);
        cls.def("columnUnit", &Table::columnUnit, "column"_a);
        cls.def("columnLenC", &Table::columnLenC, "column"_a);
        cls.def("columnShape", &Table::columnShape, "column"_a);
        cls.def_prop_ro("nColumn", &Table::getNColumn);
        cls.def_prop_ro("nRow", &Table::getNRow);
      });
}

}  // namespace ast
