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

PYBIND11_MODULE(channel, mod) {
    py::module::import("astshim.object");

    py::class_<Channel, std::shared_ptr<Channel>, Object> cls(mod, "Channel");

    cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

    cls.def_property("comment", &Channel::getComment, &Channel::setComment);
    cls.def_property("full", &Channel::getFull, &Channel::setFull);
    cls.def_property("indent", &Channel::getIndent, &Channel::setIndent);
    cls.def_property("reportLevel", &Channel::getReportLevel, &Channel::setReportLevel);
    cls.def_property("skip", &Channel::getSkip, &Channel::setSkip);
    cls.def_property("strict", &Channel::getStrict, &Channel::setStrict);

    cls.def("copy", &Channel::copy);
    cls.def("read", &Channel::read);
    cls.def("write", &Channel::write, "object"_a);
    cls.def("warnings", &Channel::warnings);
}

}  // namespace
}  // namespace ast
