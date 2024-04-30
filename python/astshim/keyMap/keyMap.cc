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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "lsst/cpputils/python.h"

#include "astshim/KeyMap.h"
#include "astshim/Object.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace ast {
void wrapKeyMap(lsst::cpputils::python::WrapperCollection &wrappers){
    using PyKeyMap =  nb::class_<KeyMap, Object> ;
    wrappers.wrapType(PyKeyMap (wrappers.module, "KeyMap"), [](auto &mod, auto &cls) {


        cls.def(nb::init<std::string const &>(), "options"_a = "");

        cls.def("copy", &KeyMap::copy);
        cls.def("defined", &KeyMap::defined, "key"_a);
        cls.def("key", &KeyMap::key, "ind"_a);
        cls.def("hasKey", &KeyMap::hasKey, "key"_a);
        cls.def("length", &KeyMap::length);
        cls.def("__len__", &KeyMap::size);

        cls.def("getD", nb::overload_cast<std::string const &, int>(&KeyMap::getD, nb::const_), "key"_a, "ind"_a);
        cls.def("getD", nb::overload_cast<std::string const &>(&KeyMap::getD, nb::const_), "key"_a);
        cls.def("getF", nb::overload_cast<std::string const &, int>(&KeyMap::getF, nb::const_), "key"_a, "ind"_a);
        cls.def("getF", nb::overload_cast<std::string const &>(&KeyMap::getF, nb::const_), "key"_a);
        cls.def("getI", nb::overload_cast<std::string const &, int>(&KeyMap::getI, nb::const_), "key"_a, "ind"_a);
        cls.def("getI", nb::overload_cast<std::string const &>(&KeyMap::getI, nb::const_), "key"_a);
        cls.def("getS", nb::overload_cast<std::string const &, int>(&KeyMap::getS, nb::const_), "key"_a, "ind"_a);
        cls.def("getS", nb::overload_cast<std::string const &>(&KeyMap::getS, nb::const_), "key"_a);
        cls.def("getB", nb::overload_cast<std::string const &, int>(&KeyMap::getB, nb::const_), "key"_a, "ind"_a);
        cls.def("getB", nb::overload_cast<std::string const &>(&KeyMap::getB, nb::const_), "key"_a);
        cls.def("getC", nb::overload_cast<std::string const &, int>(&KeyMap::getC, nb::const_), "key"_a, "ind"_a);
        cls.def("getC", nb::overload_cast<std::string const &>(&KeyMap::getC, nb::const_), "key"_a);
        cls.def("getA", nb::overload_cast<std::string const &, int>(&KeyMap::getA, nb::const_), "key"_a, "ind"_a);
        cls.def("getA", nb::overload_cast<std::string const &>(&KeyMap::getA, nb::const_), "key"_a);

        cls.def("putD", nb::overload_cast<std::string const &, double, std::string const &>(&KeyMap::putD),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putD",
                nb::overload_cast<std::string const &, std::vector<double> const &, std::string const &>(
                        &KeyMap::putD),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putF", nb::overload_cast<std::string const &, float, std::string const &>(&KeyMap::putF),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putF",
                nb::overload_cast<std::string const &, std::vector<float> const &, std::string const &>(
                        &KeyMap::putF),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putI", nb::overload_cast<std::string const &, int, std::string const &>(&KeyMap::putI), "key"_a,
                "value"_a, "comment"_a = "");
        cls.def("putI",
                nb::overload_cast<std::string const &, std::vector<int> const &, std::string const &>(
                        &KeyMap::putI),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putS", nb::overload_cast<std::string const &, short int, std::string const &>(&KeyMap::putS),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putS",
                nb::overload_cast<std::string const &, std::vector<short int> const &, std::string const &>(
                        &KeyMap::putS),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putB", nb::overload_cast<std::string const &, char unsigned, std::string const &>(&KeyMap::putB),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putB",
                nb::overload_cast<std::string const &, std::vector<char unsigned> const &, std::string const &>(
                        &KeyMap::putB),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putC",
                nb::overload_cast<std::string const &, std::string const &, std::string const &>(&KeyMap::putC),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putC",
                nb::overload_cast<std::string const &, std::vector<std::string> const &, std::string const &>(
                        &KeyMap::putC),
                "key"_a, "vec"_a, "comment"_a = "");
        cls.def("putA",
                nb::overload_cast<std::string const &, Object const &, std::string const &>(&KeyMap::putA),
                "key"_a, "value"_a, "comment"_a = "");
        cls.def("putA",
                nb::overload_cast<std::string const &, std::vector<std::shared_ptr<Object const>> const &,
                        std::string const &>(&KeyMap::putA),
                "key"_a, "vec"_a, "comment"_a = "");

        cls.def("putU", nb::overload_cast<std::string const &, std::string const &>(&KeyMap::putU), "key"_a,
                "comment"_a = "");

        // there no need to wrap float, short int and unsigned char versions since the type of the key
        // is already known and the data will be cast
        // list int first, then float because the first match is used
        cls.def("append", nb::overload_cast<std::string const &, int>(&KeyMap::append), "key"_a, "value"_a);
        cls.def("append", nb::overload_cast<std::string const &, double>(&KeyMap::append), "key"_a, "value"_a);
        cls.def("append", nb::overload_cast<std::string const &, std::string const &>(&KeyMap::append), "key"_a,
                "value"_a);
        cls.def("append", nb::overload_cast<std::string const &, Object const &>(&KeyMap::append), "key"_a,
                "value"_a);

        // there no need to wrap float, short int and unsigned char versions since the type of the key
        // is already known and the data will be cast
        // list int first, then float because the first match is used
        cls.def("replace", nb::overload_cast<std::string const &, int, int>(&KeyMap::replace), "key"_a, "ind"_a,
                "value"_a);
        cls.def("replace", nb::overload_cast<std::string const &, int, double>(&KeyMap::replace), "key"_a,
                "ind"_a, "value"_a);
        cls.def("replace", nb::overload_cast<std::string const &, int, std::string const &>(&KeyMap::replace),
                "key"_a, "ind"_a, "value"_a);
        cls.def("replace", nb::overload_cast<std::string const &, int, Object const &>(&KeyMap::replace), "key"_a,
                "ind"_a, "value"_a);

        cls.def("remove", &KeyMap::remove, "key"_a);
        cls.def("rename", &KeyMap::rename, "oldKey"_a, "newKey"_a);
        cls.def("type", &KeyMap::type, "key"_a);
    });
}
}  // namespace ast
