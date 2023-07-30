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
#include <complex>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "lsst/cpputils/python.h"

#include "astshim/Channel.h"
#include "astshim/FitsChan.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

    template<typename T>
    void wrapFoundValue(lsst::utils::python::WrapperCollection &wrappers, std::string const &suffix) {
        using PyFoundValue = py::class_<FoundValue<T>>;
        std::string name = "FoundValue" + suffix;
        wrappers.wrapType(PyFoundValue(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
            cls.def(py::init<bool, T const &>(), "found"_a, "value"_a);
            cls.def_readwrite("found", &FoundValue<T>::found);
            cls.def_readwrite("value", &FoundValue<T>::value);
        });
    }
} // namespace

void wrapFitsChan(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::enum_<FitsKeyState>(wrappers.module, "FitsKeyState"), [](auto &mod, auto &enm) {
        enm.value("ABSENT", FitsKeyState::ABSENT);
        enm.value("NOVALUE", FitsKeyState::NOVALUE);
        enm.value("PRESENT", FitsKeyState::PRESENT);
        enm.export_values();
    });

    wrappers.wrapType(py::enum_<CardType>(wrappers.module, "CardType"), [](auto &mod, auto &enm) {
        enm.value("NOTYPE", CardType::NOTYPE);
        enm.value("COMMENT", CardType::COMMENT);
        enm.value("INT", CardType::INT);
        enm.value("FLOAT", CardType::FLOAT);
        enm.value("STRING", CardType::STRING);
        enm.value("COMPLEXF", CardType::COMPLEXF);
        enm.value("COMPLEXI", CardType::COMPLEXI);
        enm.value("LOGICAL", CardType::LOGICAL);
        enm.value("CONTINUE", CardType::CONTINUE);
        enm.value("UNDEF", CardType::UNDEF);
        enm.export_values();
    });

    // Wrap FoundValue struct for returning various types
    wrapFoundValue<std::string>(wrappers, "S");
    wrapFoundValue<std::complex<double>>(wrappers, "CF");
    wrapFoundValue<double>(wrappers, "F");
    wrapFoundValue<int>(wrappers, "I");
    wrapFoundValue<bool>(wrappers, "L");

    // Wrap FitsChan
    using PyFitsChan =  py::class_<FitsChan, Channel>;
    wrappers.wrapType(PyFitsChan(wrappers.module, "FitsChan"), [](auto &mod, auto &cls) {
        cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

        cls.def_property("carLin", &FitsChan::getCarLin, &FitsChan::setCarLin);
        cls.def_property("cdMatrix", &FitsChan::getCDMatrix, &FitsChan::setCDMatrix);
        cls.def_property("clean", &FitsChan::getClean, &FitsChan::setClean);
        cls.def_property("defB1950", &FitsChan::getDefB1950, &FitsChan::setDefB1950);
        cls.def_property("encoding", &FitsChan::getEncoding, &FitsChan::setEncoding);
        cls.def_property("fitsAxisOrder", &FitsChan::getFitsAxisOrder, &FitsChan::setFitsAxisOrder);
        cls.def_property("fitsDigits", &FitsChan::getFitsDigits, &FitsChan::setFitsDigits);
        cls.def_property_readonly("nCard", &FitsChan::getNCard);
        cls.def_property_readonly("nKey", &FitsChan::getNKey);
        cls.def_property("iwc", &FitsChan::getIwc, &FitsChan::setIwc);
        cls.def_property("sipOK", &FitsChan::getSipOK, &FitsChan::setSipOK);
        cls.def_property("sipReplace", &FitsChan::getSipReplace, &FitsChan::setSipReplace);
        cls.def_property("tabOK", &FitsChan::getTabOK, &FitsChan::setTabOK);
        cls.def_property("polyTan", &FitsChan::getPolyTan, &FitsChan::setPolyTan);
        cls.def_property("warnings", &FitsChan::getWarnings, &FitsChan::setWarnings);
        cls.def_property("fitsTol", &FitsChan::getFitsTol, &FitsChan::setFitsTol);

        cls.def("delFits", &FitsChan::delFits);
        cls.def("emptyFits", &FitsChan::emptyFits);
        cls.def("findFits", &FitsChan::findFits, "name"_a, "inc"_a);
        cls.def("getFitsCF", &FitsChan::getFitsCF, "name"_a = "", "defval"_a = std::complex<double>(0, 0));
        cls.def("getFitsCN", &FitsChan::getFitsCN, "name"_a = "", "defval"_a = "");
        cls.def("getFitsF", &FitsChan::getFitsF, "name"_a = "", "defval"_a = 0);
        cls.def("getFitsI", &FitsChan::getFitsI, "name"_a = "", "defval"_a = 0);
        cls.def("getFitsL", &FitsChan::getFitsL, "name"_a = "", "defval"_a = false);
        cls.def("getFitsS", &FitsChan::getFitsS, "name"_a = "", "defval"_a = "");
        cls.def("getAllCardNames", &FitsChan::getAllCardNames);
        cls.def("getAllWarnings", &FitsChan::getAllWarnings);
        cls.def("getCard", &FitsChan::getCard);
        cls.def("getCardComm", &FitsChan::getCardComm);
        cls.def("getCardName", &FitsChan::getCardName);
        cls.def("getCardType", &FitsChan::getCardType);
        cls.def("getTables", &FitsChan::getTables);
        cls.def("purgeWcs", &FitsChan::purgeWcs);
        cls.def("putCards", &FitsChan::putCards, "cards"_a);
        cls.def("putFits", &FitsChan::putFits, "card"_a, "overwrite"_a);
        cls.def("readFits", &FitsChan::readFits);
        cls.def("retainFits", &FitsChan::retainFits);
        cls.def("setFitsCF", &FitsChan::setFitsCF, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsCM", &FitsChan::setFitsCM, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsCN", &FitsChan::setFitsCN, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsF", &FitsChan::setFitsF, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsI", &FitsChan::setFitsI, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsL", &FitsChan::setFitsL, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsS", &FitsChan::setFitsS, "name"_a, "value"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("setFitsU", &FitsChan::setFitsU, "name"_a, "comment"_a = "", "overwrite"_a = false);
        cls.def("showFits", &FitsChan::showFits);
        cls.def("testFits", &FitsChan::testFits, "name"_a = "");
        cls.def("writeFits", &FitsChan::writeFits);
        cls.def("clearCard", &FitsChan::clearCard);
        cls.def("setCard", &FitsChan::setCard, "i"_a);
    });
}

}  // namespace ast
