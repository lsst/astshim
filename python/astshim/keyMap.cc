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
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "astshim/base.h"
#include "astshim/KeyMap.h"
#include "astshim/Object.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

PYBIND11_PLUGIN(keyMap) {
    py::module::import("astshim.object");

    py::module mod("keyMap", "Python wrapper for KeyMap");

    py::class_<KeyMap, std::shared_ptr<KeyMap>, Object> cls(mod, "KeyMap");

    cls.def(py::init<std::string const &>(), "options"_a="");

    cls.def("copy", &KeyMap::copy);
    cls.def("defined", &KeyMap::defined, "key"_a);
    cls.def("key", &KeyMap::key, "ind"_a);
    cls.def("hasKey", &KeyMap::hasKey, "key"_a);
    cls.def("length", &KeyMap::length);
    cls.def("__len__", &KeyMap::size);

    cls.def("getD", (double (KeyMap::*)(std::string const &, int) const) &KeyMap::getD, "key"_a, "ind"_a);
    cls.def("getD", (std::vector<double> (KeyMap::*)(std::string const &) const) &KeyMap::getD, "key"_a);
    cls.def("getF", (float (KeyMap::*)(std::string const &, int) const) &KeyMap::getF, "key"_a, "ind"_a);
    cls.def("getF", (std::vector<float> (KeyMap::*)(std::string const &) const) &KeyMap::getF, "key"_a);
    cls.def("getI", (int (KeyMap::*)(std::string const &, int) const) &KeyMap::getI, "key"_a, "ind"_a);
    cls.def("getI", (std::vector<int> (KeyMap::*)(std::string const &) const) &KeyMap::getI, "key"_a);
    cls.def("getS", (short int (KeyMap::*)(std::string const &, int) const) &KeyMap::getS, "key"_a, "ind"_a);
    cls.def("getS", (std::vector<short int> (KeyMap::*)(std::string const &) const) &KeyMap::getS, "key"_a);
    cls.def("getB", (char unsigned (KeyMap::*)(std::string const &, int) const) &KeyMap::getB, "key"_a, "ind"_a);
    cls.def("getB", (std::vector<char unsigned> (KeyMap::*)(std::string const &) const) &KeyMap::getB, "key"_a);
    cls.def("getC", (std::string (KeyMap::*)(std::string const &, int) const) &KeyMap::getC, "key"_a, "ind"_a);
    cls.def("getC", (std::vector<std::string> (KeyMap::*)(std::string const &) const) &KeyMap::getC, "key"_a);
    cls.def("getA", (std::shared_ptr<Object> (KeyMap::*)(std::string const &, int) const) &KeyMap::getA, "key"_a, "ind"_a);
    cls.def("getA", (std::vector<std::shared_ptr<Object>> (KeyMap::*)(std::string const &) const) &KeyMap::getA, "key"_a);

    cls.def("putD", (void (KeyMap::*)(std::string const &, double, std::string const &)) &KeyMap::putD, "key"_a, "value"_a, "comment"_a="");
    cls.def("putD", (void (KeyMap::*)(std::string const &, std::vector<double> const &, std::string const &)) &KeyMap::putD, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putF", (void (KeyMap::*)(std::string const &, float, std::string const &)) &KeyMap::putF, "key"_a, "value"_a, "comment"_a="");
    cls.def("putF", (void (KeyMap::*)(std::string const &, std::vector<float> const &, std::string const &)) &KeyMap::putF, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putI", (void (KeyMap::*)(std::string const &, int, std::string const &)) &KeyMap::putI, "key"_a, "value"_a, "comment"_a="");
    cls.def("putI", (void (KeyMap::*)(std::string const &, std::vector<int> const &, std::string const &)) &KeyMap::putI, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putS", (void (KeyMap::*)(std::string const &, short int, std::string const &)) &KeyMap::putS, "key"_a, "value"_a, "comment"_a="");
    cls.def("putS", (void (KeyMap::*)(std::string const &, std::vector<short int> const &, std::string const &)) &KeyMap::putS, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putB", (void (KeyMap::*)(std::string const &, char unsigned, std::string const &)) &KeyMap::putB, "key"_a, "value"_a, "comment"_a="");
    cls.def("putB", (void (KeyMap::*)(std::string const &, std::vector<char unsigned> const &, std::string const &)) &KeyMap::putB, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putC", (void (KeyMap::*)(std::string const &, std::string const &, std::string const &)) &KeyMap::putC, "key"_a, "value"_a, "comment"_a="");
    cls.def("putC", (void (KeyMap::*)(std::string const &, std::vector<std::string const> const &, std::string const &)) &KeyMap::putC, "key"_a, "vec"_a, "comment"_a="");
    cls.def("putA", (void (KeyMap::*)(std::string const &, Object const &, std::string const &)) &KeyMap::putA, "key"_a, "value"_a, "comment"_a="");
    cls.def("putA", (void (KeyMap::*)(std::string const &, std::vector<std::shared_ptr<Object const>> const &, std::string const &)) &KeyMap::putA, "key"_a, "vec"_a, "comment"_a="");

    cls.def("putU", (void (KeyMap::*)(std::string const &, std::string const &)) &KeyMap::putU, "key"_a, "comment"_a="");


    // there no need to wrap float, short int and unsigned char versions since the type of the key
    // is already known and the data will be cast
    // list int first, then float because the first match is used
    cls.def("append", (void (KeyMap::*)(std::string const &, int)) &KeyMap::append, "key"_a, "value"_a);
    cls.def("append", (void (KeyMap::*)(std::string const &, double)) &KeyMap::append, "key"_a, "value"_a);
    cls.def("append", (void (KeyMap::*)(std::string const &, std:: string const &)) &KeyMap::append, "key"_a, "value"_a);
    cls.def("append", (void (KeyMap::*)(std::string const &, Object const &)) &KeyMap::append, "key"_a, "value"_a);

    // there no need to wrap float, short int and unsigned char versions since the type of the key
    // is already known and the data will be cast
    // list int first, then float because the first match is used
    cls.def("replace", (void (KeyMap::*)(std::string const &, int, int)) &KeyMap::replace, "key"_a, "ind"_a, "value"_a);
    cls.def("replace", (void (KeyMap::*)(std::string const &, int, double)) &KeyMap::replace, "key"_a, "ind"_a, "value"_a);
    cls.def("replace", (void (KeyMap::*)(std::string const &, int, std:: string const &)) &KeyMap::replace, "key"_a, "ind"_a, "value"_a);
    cls.def("replace", (void (KeyMap::*)(std::string const &, int, Object const &)) &KeyMap::replace, "key"_a, "ind"_a, "value"_a);

    cls.def("remove", &KeyMap::remove, "key"_a);
    cls.def("rename", &KeyMap::rename, "oldKey"_a, "newKey"_a);
    cls.def("type", &KeyMap::type, "key"_a);

    return mod.ptr();
}

}  // ast
