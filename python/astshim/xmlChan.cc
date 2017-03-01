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

#include "astshim/Channel.h"
#include "astshim/XmlChan.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

PYBIND11_PLUGIN(xmlChan) {
    py::module::import("astshim.channel");

    py::module mod("xmlChan", "Python wrapper for XmlChan");

    py::class_<XmlChan, std::shared_ptr<XmlChan>, Channel> cls(mod, "XmlChan");

    cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a="");

    cls.def("getXmlFormat", &XmlChan::getXmlFormat);
    cls.def("getXmlLength", &XmlChan::getXmlLength);
    cls.def("getXmlPrefix", &XmlChan::getXmlPrefix);
    cls.def("setXmlFormat", &XmlChan::setXmlFormat, "format"_a);
    cls.def("setXmlLength", &XmlChan::setXmlLength, "len"_a);
    cls.def("setXmlPrefix", &XmlChan::setXmlPrefix, "prefix"_a);

    return mod.ptr();
}

}  // ast
