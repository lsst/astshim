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


#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>

#include "lsst/cpputils/python.h"

#include "astshim/Channel.h"
#include "astshim/FitsChan.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {
namespace {

    template<typename T>
    void wrapFoundValue(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
        using PyFoundValue = nb::class_<FoundValue<T>>;
        std::string name = "FoundValue" + suffix;
        wrappers.wrapType(PyFoundValue(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
            cls.def(nb::init<bool, T const &>(), "found"_a, "value"_a);
            cls.def_prop_rw("found", [](FoundValue<T> const &self) {return self.found;}, [](FoundValue<T> &self) {return self.found;});
            cls.def_prop_rw("value", [](FoundValue<T> const &self) {return self.value;}, [](FoundValue<T> &self) {return self.value;});
        });
    }
} // namespace

void wrapFitsChan(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(nb::enum_<FitsKeyState>(wrappers.module, "FitsKeyState"), [](auto &mod, auto &enm) {
        enm.value("ABSENT", FitsKeyState::ABSENT);
        enm.value("NOVALUE", FitsKeyState::NOVALUE);
        enm.value("PRESENT", FitsKeyState::PRESENT);
        enm.export_values();
    });

    wrappers.wrapType(nb::enum_<CardType>(wrappers.module, "CardType"), [](auto &mod, auto &enm) {
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
    using PyFitsChan =  nb::class_<FitsChan, Channel>;
    wrappers.wrapType(PyFitsChan(wrappers.module, "FitsChan"), [](auto &mod, auto &cls) {
        cls.def(nb::init<Stream &, std::string const &>(), "stream"_a, "options"_a = "");

        cls.def_prop_rw("carLin", &FitsChan::getCarLin, &FitsChan::setCarLin);
        cls.def_prop_rw("cdMatrix", &FitsChan::getCDMatrix, &FitsChan::setCDMatrix);
        cls.def_prop_rw("clean", &FitsChan::getClean, &FitsChan::setClean);
        cls.def_prop_rw("defB1950", &FitsChan::getDefB1950, &FitsChan::setDefB1950);
        cls.def_prop_rw("encoding", &FitsChan::getEncoding, &FitsChan::setEncoding);
        cls.def_prop_rw("fitsAxisOrder", &FitsChan::getFitsAxisOrder, &FitsChan::setFitsAxisOrder);
        cls.def_prop_rw("fitsDigits", &FitsChan::getFitsDigits, &FitsChan::setFitsDigits);
        cls.def_prop_ro("nCard", &FitsChan::getNCard);
        cls.def_prop_ro("nKey", &FitsChan::getNKey);
        cls.def_prop_rw("iwc", &FitsChan::getIwc, &FitsChan::setIwc);
        cls.def_prop_rw("sipOK", &FitsChan::getSipOK, &FitsChan::setSipOK);
        cls.def_prop_rw("sipReplace", &FitsChan::getSipReplace, &FitsChan::setSipReplace);
        cls.def_prop_rw("tabOK", &FitsChan::getTabOK, &FitsChan::setTabOK);
        cls.def_prop_rw("polyTan", &FitsChan::getPolyTan, &FitsChan::setPolyTan);
        cls.def_prop_rw("warnings", &FitsChan::getWarnings, &FitsChan::setWarnings);
        cls.def_prop_rw("fitsTol", &FitsChan::getFitsTol, &FitsChan::setFitsTol);

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
