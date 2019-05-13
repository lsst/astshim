/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
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
#include <string>
#include <vector>

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Stream.h"
#include "astshim/FitsChan.h"

namespace ast {

namespace {

/**
 * Return a C string, or nullptr if str is empty
 */
char const *cstrOrNull(std::string const &str) { return str.empty() ? nullptr : str.c_str(); }

}  // namespace

FitsChan::FitsChan(Stream &stream, std::string const &options)
        : Channel(reinterpret_cast<AstChannel *>(
                          astFitsChan(detail::source, detail::sink, "%s", options.c_str())),
                  stream, true) {
    assertOK();
}

FitsChan::~FitsChan() {
    // when an astFitsChan is destroyed it first writes out any cards, but if I let astFitsChan
    // do this automatically then it occurs while the Channel and its Source are being destroyed,
    // which is too late
    astWriteFits(getRawPtr());
    // No 'assertOK' here: can't throw in a Dtor.
}

FoundValue<std::complex<double>> FitsChan::getFitsCF(std::string const &name,
                                                     std::complex<double> defval) const {
    std::complex<double> val = defval;
    // this use of reinterpret_cast is explicitly permitted, for C compatibility
    double *rawval = reinterpret_cast<double(&)[2]>(val);
    bool found = astGetFitsCF(getRawPtr(), cstrOrNull(name), rawval);
    assertOK();
    return FoundValue<std::complex<double>>(found, val);
}

FoundValue<std::string> FitsChan::getFitsCN(std::string const &name, std::string defval) const {
    char *rawval;  // astGetFitsCN has its own static buffer for the value
    bool found = astGetFitsCN(getRawPtr(), cstrOrNull(name), &rawval);
    assertOK();
    std::string val = found ? rawval : defval;
    return FoundValue<std::string>(found, val);
}

FoundValue<double> FitsChan::getFitsF(std::string const &name, double defval) const {
    double val = defval;
    bool found = astGetFitsF(getRawPtr(), cstrOrNull(name), &val);
    assertOK();
    return FoundValue<double>(found, val);
}

FoundValue<int> FitsChan::getFitsI(std::string const &name, int defval) const {
    int val = defval;
    bool found = astGetFitsI(getRawPtr(), cstrOrNull(name), &val);
    assertOK();
    return FoundValue<int>(found, val);
}

FoundValue<bool> FitsChan::getFitsL(std::string const &name, bool defval) const {
    int val = static_cast<int>(defval);
    bool found = astGetFitsL(getRawPtr(), cstrOrNull(name), &val);
    assertOK();
    return FoundValue<bool>(found, static_cast<bool>(val));
}

FoundValue<std::string> FitsChan::getFitsS(std::string const &name, std::string defval) const {
    char *rawval;  // astGetFitsS has its own static buffer for the value
    bool found = astGetFitsS(getRawPtr(), cstrOrNull(name), &rawval);
    assertOK();
    std::string val = found ? rawval : defval;
    return FoundValue<std::string>(found, val);
}

std::vector<std::string> FitsChan::getAllCardNames() {
    int const initialIndex = getCard();
    int const numCards = getNCard();
    std::vector<std::string> nameList;
    nameList.reserve(numCards);
    try {
        for (auto i = 1; i <= numCards; ++i) {
            setCard(i);
            nameList.emplace_back(getCardName());
        }
    } catch (...) {
        setCard(initialIndex);
        throw;
    }
    setCard(initialIndex);
    return nameList;
}

FoundValue<std::string> FitsChan::findFits(std::string const &name, bool inc) {
    std::unique_ptr<char[]> fitsbuf(new char[detail::FITSLEN + 1]);
    fitsbuf[0] = '\0';  // in case nothing is found
    bool success = static_cast<bool>(astFindFits(getRawPtr(), name.c_str(), fitsbuf.get(), inc));
    assertOK();
    return FoundValue<std::string>(success, std::string(fitsbuf.get()));
}

FitsKeyState FitsChan::testFits(std::string const &name) const {
    int there;
    int hasvalue = astTestFits(getRawPtr(), cstrOrNull(name), &there);
    assertOK();
    if (hasvalue) {
        return FitsKeyState::PRESENT;
    }
    return there ? FitsKeyState::NOVALUE : FitsKeyState::ABSENT;
}

}  // namespace ast
