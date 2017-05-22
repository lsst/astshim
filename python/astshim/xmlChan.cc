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
namespace {

PYBIND11_PLUGIN(xmlChan) {
    py::module mod("xmlChan", "Python wrapper for XmlChan");

    py::module::import("astshim.channel");

    py::class_<XmlChan, std::shared_ptr<XmlChan>, Channel> cls(mod, "XmlChan");

    cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

    cls.def_property("xmlFormat", &XmlChan::getXmlFormat, &XmlChan::setXmlFormat);
    cls.def_property("xmlLength", &XmlChan::getXmlLength, &XmlChan::setXmlLength);
    cls.def_property("xmlPrefix", &XmlChan::getXmlPrefix, &XmlChan::setXmlPrefix);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
