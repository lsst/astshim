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
#include <nanobind/stl/shared_ptr.h>
#include "lsst/cpputils/python.h"

#include "ndarray/nanobind.h"
#include "astshim/base.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapBase(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.module.def("assertOK", &assertOK, "rawObj1"_a = nullptr, "rawObj2"_a = nullptr);
    wrappers.module.def("escapes", &escapes, "include"_a = -1);
    wrappers.module.def("astVersion", &ast_version);

    // Make a deep copy to avoid memory issues in Python
    wrappers.module.def("arrayFromVector", [](std::vector<double> const& data, int nAxes) {
        auto const arrayShallow = arrayFromVector(data, nAxes);
        Array2D arrayDeep = allocate(arrayShallow.getShape());
        arrayDeep.deep() = arrayShallow;
        return arrayDeep;
    }, "vec"_a, "nAxes"_a);

    wrappers.wrapType(nb::enum_<DataType>(wrappers.module, "DataType"), [](auto &mod, auto &enm) {
        enm.value("IntType", DataType::IntType);
        enm.value("ShortIntType", DataType::ShortIntType);
        enm.value("ByteType", DataType::ByteType);
        enm.value("DoubleType", DataType::DoubleType);
        enm.value("FloatType", DataType::FloatType);
        enm.value("StringType", DataType::StringType);
        enm.value("ObjectType", DataType::ObjectType);
        enm.value("PointerType", DataType::PointerType);
        enm.value("UndefinedType", DataType::UndefinedType);
        enm.value("BadType", DataType::BadType);
        enm.export_values();
    });
}

}  // namespace ast
