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

#include "astshim/Mapping.h"
#include "astshim/WcsMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(wcsMap) {
    py::module mod("wcsMap", "Python wrapper for WcsMap");

    py::module::import("astshim.mapping");

    py::enum_<WcsType>(mod, "WcsType")
        .value("AZP", WcsType::AZP)
        .value("SZP", WcsType::SZP)
        .value("TAN", WcsType::TAN)
        .value("STG", WcsType::STG)
        .value("SIN", WcsType::SIN)
        .value("ARC", WcsType::ARC)
        .value("ZPN", WcsType::ZPN)
        .value("ZEA", WcsType::ZEA)
        .value("AIR", WcsType::AIR)
        .value("CYP", WcsType::CYP)
        .value("CEA", WcsType::CEA)
        .value("CAR", WcsType::CAR)
        .value("MER", WcsType::MER)
        .value("SFL", WcsType::SFL)
        .value("PAR", WcsType::PAR)
        .value("MOL", WcsType::MOL)
        .value("AIT", WcsType::AIT)
        .value("COP", WcsType::COP)
        .value("COE", WcsType::COE)
        .value("COD", WcsType::COD)
        .value("COO", WcsType::COO)
        .value("BON", WcsType::BON)
        .value("PCO", WcsType::PCO)
        .value("TSC", WcsType::TSC)
        .value("CSC", WcsType::CSC)
        .value("QSC", WcsType::QSC)
        .value("NCP", WcsType::NCP)
        .value("GLS", WcsType::GLS)
        .value("TPN", WcsType::TPN)
        .value("HPX", WcsType::HPX)
        .value("XPH", WcsType::XPH)
        .value("WCSBAD", WcsType::WCSBAD)
        .export_values();

    py::class_<WcsMap, std::shared_ptr<WcsMap>, Mapping> cls(mod, "WcsMap");

    cls.def(py::init<int, WcsType, int, int, std::string const &>(),
            "ncoord"_a, "type"_a, "lonax"_a, "latax"_a, "options"_a="");

    cls.def("copy", &WcsMap::copy);
    cls.def("getNatLat", &WcsMap::getNatLat);
    cls.def("getNatLon", &WcsMap::getNatLon);
    cls.def("getPVi_m", &WcsMap::getPVi_m, "i"_a, "m"_a);
    cls.def("getPVMax", &WcsMap::getPVMax);
    cls.def("getWcsAxis", &WcsMap::getWcsAxis);
    cls.def("getWcsType", &WcsMap::getWcsType);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
