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

#include "astshim/base.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

PYBIND11_PLUGIN(base) {
    py::module mod("base", "Python wrapper for base.h");

    // Note: do not wrap arrayFromVector because it is unsafe (the array
    // will be corrupted if the vector is deleted) and unnecessary (use numpy)
    mod.def("assertOK", &assertOK, "rawObj1"_a=nullptr, "rawObj2"_a=nullptr);
    mod.def("escapes", &escapes, "include"_a=-1);

    py::enum_<DataType>(mod, "DataType")
        .value("IntType", DataType::IntType)
        .value("ShortIntType", DataType::ShortIntType)
        .value("ByteType", DataType::ByteType)
        .value("DoubleType", DataType::DoubleType)
        .value("FloatType", DataType::FloatType)
        .value("StringType", DataType::StringType)
        .value("ObjectType", DataType::ObjectType)
        .value("PointerType", DataType::PointerType)
        .value("UndefinedType", DataType::UndefinedType)
        .value("BadType", DataType::BadType)
        .export_values();


    return mod.ptr();
}

}  // ast
