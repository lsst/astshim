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
#include "lsst/cpputils/python.h"

#include "astshim/Channel.h"
#include "astshim/KeyMap.h"
#include "astshim/Object.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapChannel(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyChannel = nb::class_<Channel, Object>;
    wrappers.wrapType(PyChannel(wrappers.module, "Channel"), [](auto &mod, auto &cls) {

        cls.def(nb::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

        cls.def_prop_rw("comment", &Channel::getComment, &Channel::setComment);
        cls.def_prop_rw("full", &Channel::getFull, &Channel::setFull);
        cls.def_prop_rw("indent", &Channel::getIndent, &Channel::setIndent);
        cls.def_prop_rw("reportLevel", &Channel::getReportLevel, &Channel::setReportLevel);
        cls.def_prop_rw("skip", &Channel::getSkip, &Channel::setSkip);
        cls.def_prop_rw("strict", &Channel::getStrict, &Channel::setStrict);

        cls.def("copy", &Channel::copy);
        cls.def("read", &Channel::read);
        cls.def("write", &Channel::write, "object"_a);
        cls.def("warnings", &Channel::warnings);
    });
}

}  // namespace ast
