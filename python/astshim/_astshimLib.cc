/*
 * This file is part of astshim.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11/pybind11.h"
#include "lsst/cpputils/python.h"

namespace py = pybind11;
using namespace pybind11::literals;
using lsst::cpputils::python::WrapperCollection;

namespace ast {
void wrapBase(WrapperCollection&);
void wrapChannel(WrapperCollection&);
void wrapChebyMap(WrapperCollection&);
void wrapCmpFrame(WrapperCollection&);
void wrapCmpMap(WrapperCollection&);
void wrapFitsChan(WrapperCollection&);
void wrapFitsTable(WrapperCollection&);
void wrapFrame(WrapperCollection&);
void wrapFrameDict(WrapperCollection&);
void wrapFrameSet(WrapperCollection&);
void wrapFunctional(WrapperCollection&);
void wrapKeyMap(WrapperCollection&);
void wrapLutMap(WrapperCollection&);
void wrapMapBox(WrapperCollection&);
void wrapMapping(WrapperCollection&);
void wrapMapSplit(WrapperCollection&);
void wrapMathMap(WrapperCollection&);
void wrapMatrixMap(WrapperCollection&);
void wrapNormMap(WrapperCollection&);
void wrapObject(WrapperCollection&);
void wrapParallelMap(WrapperCollection&);
void wrapPcdMap(WrapperCollection&);
void wrapPermMap(WrapperCollection&);
void wrapPolyMap(WrapperCollection&);
void wrapQuadApprox(WrapperCollection&);
void wrapRateMap(WrapperCollection&);
void wrapSeriesMap(WrapperCollection&);
void wrapShiftMap(WrapperCollection&);
void wrapSkyFrame(WrapperCollection&);
void wrapSlaMap(WrapperCollection&);
void wrapSpecFrame(WrapperCollection&);
void wrapSphMap(WrapperCollection&);
void wrapStream(WrapperCollection&);
void wrapTable(WrapperCollection&);
void wrapTimeFrame(WrapperCollection&);
void wrapTimeMap(WrapperCollection&);
void wrapTranMap(WrapperCollection&);
void wrapUnitMap(WrapperCollection&);
void wrapUnitNormMap(WrapperCollection&);
void wrapWcsMap(WrapperCollection&);
void wrapWinMap(WrapperCollection&);
void wrapXmlChan(WrapperCollection&);
void wrapZoomMap(WrapperCollection&);

PYBIND11_MODULE(_astshimLib, mod) {
    using lsst::cpputils::python::WrapperCollection;
    lsst::cpputils::python::WrapperCollection wrappers(mod, "astshim");
    wrapBase(wrappers);
    wrapObject(wrappers);
    wrapKeyMap(wrappers);
    wrapMapping(wrappers);
    wrapTable(wrappers);
    wrapChannel(wrappers);
    wrapCmpMap(wrappers);
    wrapFrame(wrappers);
    wrapFrameSet(wrappers);
    wrapMapBox(wrappers);
    wrapMapSplit(wrappers);

    wrapChebyMap(wrappers);
    wrapCmpFrame(wrappers);
    wrapFitsChan(wrappers);
    wrapFitsTable(wrappers);
    wrapFrameDict(wrappers);
    wrapFunctional(wrappers);
    wrapLutMap(wrappers);
    wrapMathMap(wrappers);
    wrapMatrixMap(wrappers);
    wrapNormMap(wrappers);
    wrapParallelMap(wrappers);
    wrapPcdMap(wrappers);
    wrapPermMap(wrappers);
    wrapPolyMap(wrappers);
    wrapQuadApprox(wrappers);
    wrapRateMap(wrappers);
    wrapSeriesMap(wrappers);
    wrapShiftMap(wrappers);
    wrapSkyFrame(wrappers);
    wrapSlaMap(wrappers);
    wrapSpecFrame(wrappers);
    wrapSphMap(wrappers);
    wrapStream(wrappers);
    wrapTimeFrame(wrappers);
    wrapTimeMap(wrappers);
    wrapTranMap(wrappers);
    wrapUnitMap(wrappers);
    wrapUnitNormMap(wrappers);
    wrapWcsMap(wrappers);
    wrapWinMap(wrappers);
    wrapXmlChan(wrappers);
    wrapZoomMap(wrappers);
    wrappers.finish();
}

}  // namespace astshim
