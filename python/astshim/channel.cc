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
//#include <pybind11/stl.h>

#include "astshim/Channel.h"
#include "astshim/KeyMap.h"
#include "astshim/Object.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(channel) {
    py::module mod("channel", "Python wrapper for Channel");

    py::module::import("astshim.object");

    py::class_<Channel, std::shared_ptr<Channel>, Object> cls(mod, "Channel");

    cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a="");

    cls.def("copy", &Channel::copy);
    cls.def("getComment", &Channel::getComment);
    cls.def("getFull", &Channel::getFull);
    cls.def("getIndent", &Channel::getIndent);
    cls.def("getReportLevel", &Channel::getReportLevel);
    cls.def("getSkip", &Channel::getSkip);
    cls.def("getStrict", &Channel::getStrict);
    cls.def("read", &Channel::read);
    cls.def("setComment", &Channel::setComment, "skip"_a);
    cls.def("setFull", &Channel::setFull, "full"_a);
    cls.def("setIndent", &Channel::setIndent, "indent"_a);
    cls.def("setReportLevel", &Channel::setReportLevel, "level"_a);
    cls.def("setSkip", &Channel::setSkip, "skip"_a);
    cls.def("setStrict", &Channel::setStrict, "strict"_a);
    cls.def("write", &Channel::write, "object"_a);
    cls.def("warnings", &Channel::warnings);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
