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
#include "lsst/cpputils/python.h"

#include "astshim/Mapping.h"
#include "astshim/WcsMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapWcsMap(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyWcsType=py::enum_<WcsType>;
    wrappers.wrapType(PyWcsType(wrappers.module, "WcsType"), [](auto &mod, auto &enm) {
        enm.value("AZP", WcsType::AZP);
        enm.value("SZP", WcsType::SZP);
        enm.value("TAN", WcsType::TAN);
        enm.value("STG", WcsType::STG);
        enm.value("SIN", WcsType::SIN);
        enm.value("ARC", WcsType::ARC);
        enm.value("ZPN", WcsType::ZPN);
        enm.value("ZEA", WcsType::ZEA);
        enm.value("AIR", WcsType::AIR);
        enm.value("CYP", WcsType::CYP);
        enm.value("CEA", WcsType::CEA);
        enm.value("CAR", WcsType::CAR);
        enm.value("MER", WcsType::MER);
        enm.value("SFL", WcsType::SFL);
        enm.value("PAR", WcsType::PAR);
        enm.value("MOL", WcsType::MOL);
        enm.value("AIT", WcsType::AIT);
        enm.value("COP", WcsType::COP);
        enm.value("COE", WcsType::COE);
        enm.value("COD", WcsType::COD);
        enm.value("COO", WcsType::COO);
        enm.value("BON", WcsType::BON);
        enm.value("PCO", WcsType::PCO);
        enm.value("TSC", WcsType::TSC);
        enm.value("CSC", WcsType::CSC);
        enm.value("QSC", WcsType::QSC);
        enm.value("NCP", WcsType::NCP);
        enm.value("GLS", WcsType::GLS);
        enm.value("TPN", WcsType::TPN);
        enm.value("HPX", WcsType::HPX);
        enm.value("XPH", WcsType::XPH);
        enm.value("WCSBAD", WcsType::WCSBAD);
        enm.export_values();
    });

    using PyWcsMap= py::class_<WcsMap, std::shared_ptr<WcsMap>, Mapping>;
    wrappers.wrapType(PyWcsMap (wrappers.module, "WcsMap"), [](auto &mod, auto &cls) {

        cls.def(py::init<int, WcsType, int, int, std::string const &>(), "ncoord"_a, "type"_a, "lonax"_a,
                "latax"_a, "options"_a = "");
        cls.def(py::init<WcsMap const &>());

        cls.def_property_readonly("natLat", &WcsMap::getNatLat);
        cls.def_property_readonly("natLon", &WcsMap::getNatLon);
        cls.def_property_readonly("wcsType", &WcsMap::getWcsType);
        cls.def_property_readonly("wcsAxis", &WcsMap::getWcsAxis);

        cls.def("copy", &WcsMap::copy);
        cls.def("getPVi_m", &WcsMap::getPVi_m, "i"_a, "m"_a);
        cls.def("getPVMax", &WcsMap::getPVMax);
    });
}

}  // namespace ast
