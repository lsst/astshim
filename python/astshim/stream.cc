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
#include "lsst/cpputils/python.h"

#include "astshim/Stream.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapStream(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyStream = nb::class_<Stream>;
    wrappers.wrapType(PyStream(wrappers.module, "Stream"), [](auto &mod, auto &clsStream) {

        clsStream.def(nb::init<std::istream *, std::ostream *>(), "istream"_a, "ostream"_a);
        clsStream.def(nb::init<>());

        clsStream.def_prop_ro("isFits", &Stream::getIsFits);
        clsStream.def_prop_ro("hasStdStream", &Stream::hasStdStream);

        clsStream.def("source", &Stream::source);
        clsStream.def("sink", &Stream::sink, "str"_a);
    });

    using PyFileStream = nb::class_<FileStream, Stream>;
    wrappers.wrapType(PyFileStream(wrappers.module, "FileStream"), [](auto &mod, auto &clsFileStream) {
        clsFileStream.def(nb::init<std::string const &, bool>(), "path"_a, "doWrite"_a = false);

        clsFileStream.def_prop_ro("path", &FileStream::getPath);
    });

    using PyStringStream = nb::class_<StringStream, Stream>;
    wrappers.wrapType(PyStringStream(wrappers.module, "StringStream"), [](auto &mod, auto &clsStringStream) {
        clsStringStream.def(nb::init<std::string const &>(), "data"_a = "");

        clsStringStream.def("getSourceData", &StringStream::getSourceData);
        clsStringStream.def("getSinkData", &StringStream::getSinkData);
        clsStringStream.def("sinkToSource", &StringStream::sinkToSource);
    });
}

}  // namespace ast
