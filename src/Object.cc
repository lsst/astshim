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
#include <algorithm>
#include <functional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Object.h"
#include "astshim/Channel.h"
#include "astshim/ChebyMap.h"
#include "astshim/CmpFrame.h"
#include "astshim/FitsChan.h"
#include "astshim/FitsTable.h"
#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/FrameDict.h"
#include "astshim/KeyMap.h"
#include "astshim/LutMap.h"
#include "astshim/MathMap.h"
#include "astshim/MatrixMap.h"
#include "astshim/NormMap.h"
#include "astshim/ParallelMap.h"
#include "astshim/PcdMap.h"
#include "astshim/PermMap.h"
#include "astshim/PolyMap.h"
#include "astshim/RateMap.h"
#include "astshim/SeriesMap.h"
#include "astshim/ShiftMap.h"
#include "astshim/SkyFrame.h"
#include "astshim/SlaMap.h"
#include "astshim/SpecFrame.h"
#include "astshim/SphMap.h"
#include "astshim/Table.h"
#include "astshim/TimeFrame.h"
#include "astshim/TimeMap.h"
#include "astshim/TranMap.h"
#include "astshim/UnitMap.h"
#include "astshim/UnitNormMap.h"
#include "astshim/WcsMap.h"
#include "astshim/WinMap.h"
#include "astshim/ZoomMap.h"

namespace ast {

namespace {

/**
C function to sink data to an ostream

This function uses the macro astChannelData as thread-safe way to retrieve a pointer to the ostream.
As such, code using this function must call `astPutChannelData(ch, &os)` to save a pointer
to the ostream `os` in the channel `ch` before calling `astWrite(ch, obj)`.
*/
extern "C" void sinkToOstream(const char *text) {
    auto osptr = reinterpret_cast<std::ostream *>(astChannelData);
    (*osptr) << text << std::endl;
}

}  // anonymous namespace

bool Object::operator==(Object const &rhs) const {
    auto thisStr = this->show(false);
    auto rhsStr = rhs.show(false);
    return rhsStr == thisStr;
}

std::shared_ptr<Object> Object::_basicFromAstObject(AstObject *rawObj) {
    static std::unordered_map<std::string, std::function<std::shared_ptr<Object>(AstObject *)>>
            ClassCasterMap = {
                    {"ChebyMap", makeShim<ChebyMap, AstChebyMap>},
                    {"CmpFrame", makeShim<CmpFrame, AstCmpFrame>},
                    {"FitsChan", makeShim<FitsChan, AstFitsChan>},
                    {"FitsTable", makeShim<FitsTable, AstFitsTable>},
                    {"Frame", makeShim<Frame, AstFrame>},
                    {"FrameSet", makeShim<FrameSet, AstFrameSet>},
                    {"FrameDict", makeShim<FrameDict, AstFrameSet>},
                    {"KeyMap", makeShim<KeyMap, AstKeyMap>},
                    {"LutMap", makeShim<LutMap, AstLutMap>},
                    {"MathMap", makeShim<MathMap, AstMathMap>},
                    {"MatrixMap", makeShim<MatrixMap, AstMatrixMap>},
                    {"NormMap", makeShim<NormMap, AstNormMap>},
                    {"ParallelMap", makeShim<ParallelMap, AstCmpMap>},
                    {"PcdMap", makeShim<PcdMap, AstPcdMap>},
                    {"PermMap", makeShim<PermMap, AstPermMap>},
                    {"PolyMap", makeShim<PolyMap, AstPolyMap>},
                    {"RateMap", makeShim<RateMap, AstRateMap>},
                    {"SeriesMap", makeShim<SeriesMap, AstCmpMap>},
                    {"ShiftMap", makeShim<ShiftMap, AstShiftMap>},
                    {"SkyFrame", makeShim<SkyFrame, AstSkyFrame>},
                    {"SlaMap", makeShim<SlaMap, AstSlaMap>},
                    {"SpecFrame", makeShim<SpecFrame, AstSpecFrame>},
                    {"SphMap", makeShim<SphMap, AstSphMap>},
                    {"TimeFrame", makeShim<TimeFrame, AstTimeFrame>},
                    {"Table", makeShim<Table, AstTable>},
                    {"TimeMap", makeShim<TimeMap, AstTimeMap>},
                    {"TranMap", makeShim<TranMap, AstTranMap>},
                    {"UnitMap", makeShim<UnitMap, AstUnitMap>},
                    {"UnitNormMap", makeShim<UnitNormMap, AstUnitNormMap>},
                    {"WcsMap", makeShim<WcsMap, AstWcsMap>},
                    {"WinMap", makeShim<WinMap, AstWinMap>},
                    {"ZoomMap", makeShim<ZoomMap, AstZoomMap>},
            };
    assertOK(rawObj);
    auto className = detail::getClassName(rawObj);
    auto name_caster = ClassCasterMap.find(className);
    if (name_caster == ClassCasterMap.end()) {
        astAnnul(rawObj);
        throw std::runtime_error("Class " + className + " not supported");
    }
    return name_caster->second(rawObj);
}

template <typename Class>
std::shared_ptr<Class> Object::fromAstObject(AstObject *rawObj, bool copy) {
    AstObject *rawObjCopy = rawObj;
    if (copy) {
        rawObjCopy = reinterpret_cast<AstObject *>(astCopy(rawObj));
        astAnnul(rawObj);
    }
    assertOK(rawObjCopy);

    // Make the appropriate ast shim object and dynamically cast to the desired output type
    auto retObjectBeforeCast = Object::_basicFromAstObject(rawObjCopy);
    auto retObject = std::dynamic_pointer_cast<Class>(retObjectBeforeCast);
    if (!retObject) {
        std::ostringstream os;
        os << "The component is of type " << retObject->getClassName()
           << ", which could not be cast to the desired type " << typeid(Class).name();
        throw std::runtime_error(os.str());
    }
    return retObject;
}

void Object::show(std::ostream &os, bool showComments) const {
    Stream stream(nullptr, &os);
    Channel ch(stream, showComments ? "" : "Comment=0");
    ch.write(*this);
    assertOK();
}

std::string Object::show(bool showComments) const {
    std::ostringstream os;
    show(os, showComments);
    return os.str();
}

Object::Object(AstObject *object) : _objPtr(object, &detail::annulAstObject) {
    assertOK();
    if (!object) {
        throw std::runtime_error("Null pointer");
    }
}

// Explicit instantiations
template std::shared_ptr<KeyMap> Object::fromAstObject<KeyMap>(AstObject *, bool);
template std::shared_ptr<FitsChan> Object::fromAstObject<FitsChan>(AstObject *, bool);
template std::shared_ptr<FrameSet> Object::fromAstObject<FrameSet>(AstObject *, bool);
template std::shared_ptr<Frame> Object::fromAstObject<Frame>(AstObject *, bool);
template std::shared_ptr<Mapping> Object::fromAstObject<Mapping>(AstObject *, bool);
template std::shared_ptr<Object> Object::fromAstObject<Object>(AstObject *, bool);

}  // namespace ast
