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
#include "astshim/Object.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(object) {
    py::module mod("object", "Python wrapper for Object");

    py::module::import("astshim.base");

    py::class_<Object, std::shared_ptr<Object>> cls(mod, "Object");

    cls.def_static("fromString", &Object::fromString);
    // do not wrap fromAstObject because it uses a bare AST pointer

    cls.def("__str__", &Object::getClass);
    cls.def("__repr__", (std::string(Object::*)() const) & Object::show);

    cls.def("copy", &Object::copy);
    cls.def("clear", &Object::clear, "attrib"_a);
    cls.def("hasAttribute", &Object::hasAttribute, "attrib"_a);
    cls.def("getClass", &Object::getClass);
    cls.def("getID", &Object::getID);
    cls.def("getIdent", &Object::getIdent);
    cls.def("getNobject", &Object::getNobject);
    cls.def("getObjSize", &Object::getObjSize);
    cls.def("getRefCount", &Object::getRefCount);
    cls.def("getUseDefs", &Object::getUseDefs);
    cls.def("lock", &Object::lock, "wait"_a);
    cls.def("same", &Object::same, "other"_a);
    cls.def("setID", &Object::setID, "id"_a);
    cls.def("setIdent", &Object::setIdent, "ident"_a);
    cls.def("setUseDefs", &Object::setUseDefs, "usedefs"_a);
    cls.def("show", (std::string(Object::*)() const) & Object::show);
    cls.def("test", &Object::test, "attrib"_a);
    cls.def("unlock", &Object::unlock, "report"_a = false);
    // do not wrap getRawPtr, since it returns a bare AST pointer

    return mod.ptr();
}

}  // namespace
}  // namespace ast
