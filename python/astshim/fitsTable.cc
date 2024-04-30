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

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include "lsst/cpputils/python.h"
#include "ndarray/nanobind.h"

#include "astshim/Table.h"
#include "astshim/FitsTable.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapFitsTable(lsst::cpputils::python::WrapperCollection &wrappers) {
  using PyFitsTable = nb::class_<FitsTable, Table>;
    wrappers.wrapType(PyFitsTable(wrappers.module, "FitsTable"), [](auto &mod, auto &cls) {
        cls.def(nb::init<std::string const &>(), "options"_a = "");
        cls.def(nb::init<FitsChan const &, std::string const &>(), "header"_a, "options"_a = "");
        cls.def("getTableHeader", &FitsTable::getTableHeader);
        cls.def("columnSize", &FitsTable::columnSize, "column"_a);
        cls.def("getColumnData1D", &FitsTable::getColumnData1D, "column"_a);
    });
}

}  // namespace ast
