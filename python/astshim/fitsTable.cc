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

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "lsst/cpputils/python.h"
#include "ndarray/pybind11.h"

#include "astshim/Table.h"
#include "astshim/FitsTable.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapFitsTable(lsst::cpputils::python::WrapperCollection &wrappers) {
  using PyFitsTable = py::class_<FitsTable, std::shared_ptr<FitsTable>, Table>;
    wrappers.wrapType(PyFitsTable(wrappers.module, "FitsTable"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::string const &>(), "options"_a = "");
        cls.def(py::init<FitsChan const &, std::string const &>(), "header"_a, "options"_a = "");
        cls.def("getTableHeader", &FitsTable::getTableHeader);
        cls.def("columnSize", &FitsTable::columnSize, "column"_a);
        cls.def("getColumnData1D", &FitsTable::getColumnData1D, "column"_a);
    });
}

}  // namespace ast
