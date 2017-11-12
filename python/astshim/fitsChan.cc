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

#include "astshim/Channel.h"
#include "astshim/FitsChan.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

template <typename T>
void wrapFoundValue(py::module &mod, std::string const &suffix) {
    py::class_<FoundValue<T>> cls(mod, ("FoundValue" + suffix).c_str());

    cls.def(py::init<bool, T const &>(), "found"_a, "value"_a);

    cls.def_readwrite("found", &FoundValue<T>::found);
    cls.def_readwrite("value", &FoundValue<T>::value);
}

PYBIND11_PLUGIN(fitsChan) {
    py::module mod("fitsChan", "Python wrapper for FitsChan");

    py::module::import("astshim.channel");

    py::enum_<FitsKeyState>(mod, "FitsKeyState")
            .value("ABSENT", FitsKeyState::ABSENT)
            .value("NOVALUE", FitsKeyState::NOVALUE)
            .value("PRESENT", FitsKeyState::PRESENT)
            .export_values();

    py::enum_<CardType>(mod, "CardType")
            .value("NOTYPE", CardType::NOTYPE)
            .value("COMMENT", CardType::COMMENT)
            .value("INT", CardType::INT)
            .value("FLOAT", CardType::FLOAT)
            .value("STRING", CardType::STRING)
            .value("COMPLEXF", CardType::COMPLEXF)
            .value("COMPLEXI", CardType::COMPLEXI)
            .value("COMPLEXI", CardType::COMPLEXI)
            .value("LOGICAL", CardType::LOGICAL)
            .value("CONTINUE", CardType::CONTINUE)
            .value("UNDEF", CardType::UNDEF)
            .export_values();

    // Wrap FoundValue struct for returning various types
    wrapFoundValue<std::string>(mod, "S");
    wrapFoundValue<std::complex<double>>(mod, "CF");
    wrapFoundValue<double>(mod, "F");
    wrapFoundValue<int>(mod, "I");
    wrapFoundValue<bool>(mod, "L");

    // Wrap FitsChan
    py::class_<FitsChan, std::shared_ptr<FitsChan>, Channel> cls(mod, "FitsChan");

    cls.def(py::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

    cls.def_property("clean", &FitsChan::getClean, &FitsChan::setClean);
    cls.def_property("defB1950", &FitsChan::getDefB1950, &FitsChan::setDefB1950);
    cls.def_property("encoding", &FitsChan::getEncoding, &FitsChan::setEncoding);
    cls.def_property("fitsAxisOrder", &FitsChan::getFitsAxisOrder, &FitsChan::setFitsAxisOrder);
    cls.def_property("fitsDigits", &FitsChan::getFitsDigits, &FitsChan::setFitsDigits);
    cls.def_property_readonly("nCard", &FitsChan::getNCard);
    cls.def_property_readonly("nKey", &FitsChan::getNKey);
    cls.def_property("iwc", &FitsChan::getIwc, &FitsChan::setIwc);
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
    cls.def("getCarLin", &FitsChan::getCarLin);
    cls.def("getCDMatrix", &FitsChan::getCDMatrix);
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
    cls.def("setCDMatrix", &FitsChan::setCDMatrix, "cdmatrix"_a);
    cls.def("showFits", &FitsChan::showFits);
    cls.def("testFits", &FitsChan::testFits, "name"_a = "");
    cls.def("writeFits", &FitsChan::writeFits);
    cls.def("clearCard", &FitsChan::clearCard);
    cls.def("setCard", &FitsChan::setCard, "i"_a);

    return mod.ptr();
}

}  // namespace
}  // namespace ast
