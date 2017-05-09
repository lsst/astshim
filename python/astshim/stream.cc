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
#include <iostream>
#include <memory>

#include <pybind11/pybind11.h>

#include "astshim/Stream.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(stream) {
    py::module mod("stream", "Python wrapper for Stream.h");

    // Stream
    py::class_<Stream, std::shared_ptr<Stream>> clsStream(mod, "Stream");

    clsStream.def(py::init<std::istream *, std::ostream *>(), "istream"_a, "ostream"_a);
    clsStream.def(py::init<>());

    clsStream.def("hasStdStream", &Stream::hasStdStream);
    clsStream.def("source", &Stream::source);
    clsStream.def("sink", &Stream::sink, "str"_a);
    clsStream.def("getIsFits", &Stream::getIsFits);

    // FileStream
    py::class_<FileStream, std::shared_ptr<FileStream>, Stream> clsFileStream(mod, "FileStream");

    clsFileStream.def(py::init<std::string const &, bool>(), "path"_a, "doWrite"_a = false);

    clsFileStream.def("getPath", &FileStream::getPath);

    // StringStream
    py::class_<StringStream, std::shared_ptr<StringStream>, Stream> clsStringStream(mod, "StringStream");

    clsStringStream.def(py::init<std::string const &>(), "data"_a = "");

    clsStringStream.def("getSourceData", &StringStream::getSourceData);
    clsStringStream.def("getSinkData", &StringStream::getSinkData);
    clsStringStream.def("sinkToSource", &StringStream::sinkToSource);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
