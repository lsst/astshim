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
#include "lsst/cpputils/python.h"

#include "astshim/Stream.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapStream(lsst::utils::python::WrapperCollection &wrappers) {
    using PyStream = py::class_<Stream, std::shared_ptr<Stream>>;
    wrappers.wrapType(PyStream(wrappers.module, "Stream"), [](auto &mod, auto &clsStream) {

        clsStream.def(py::init<std::istream *, std::ostream *>(), "istream"_a, "ostream"_a);
        clsStream.def(py::init<>());

        clsStream.def_property_readonly("isFits", &Stream::getIsFits);
        clsStream.def_property_readonly("hasStdStream", &Stream::hasStdStream);

        clsStream.def("source", &Stream::source);
        clsStream.def("sink", &Stream::sink, "str"_a);
    });

    using PyFileStream = py::class_<FileStream, std::shared_ptr<FileStream>, Stream>;
    wrappers.wrapType(PyFileStream(wrappers.module, "FileStream"), [](auto &mod, auto &clsFileStream) {
        clsFileStream.def(py::init<std::string const &, bool>(), "path"_a, "doWrite"_a = false);

        clsFileStream.def_property_readonly("path", &FileStream::getPath);
    });

    using PyStringStream = py::class_<StringStream, std::shared_ptr<StringStream>, Stream>;
    wrappers.wrapType(PyStringStream(wrappers.module, "StringStream"), [](auto &mod, auto &clsStringStream) {
        clsStringStream.def(py::init<std::string const &>(), "data"_a = "");

        clsStringStream.def("getSourceData", &StringStream::getSourceData);
        clsStringStream.def("getSinkData", &StringStream::getSinkData);
        clsStringStream.def("sinkToSource", &StringStream::sinkToSource);
    });
}

}  // namespace ast
