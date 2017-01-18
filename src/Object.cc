/* 
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#include <functional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Channel.h"
#include "astshim/CmpMap.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"
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

} // anonymous namespace

std::shared_ptr<Object> Object::fromAstObject(AstObject * rawObj) {
    static std::unordered_map<std::string,
                          std::function<std::shared_ptr<Object>(AstObject *)>> ClassCasterMap = {
        {"Channel", makeShim<Channel, AstChannel>},
        {"CmpMap", makeShim<CmpMap, AstCmpMap>},
        {"Mapping", makeShim<Mapping, AstMapping>},
        {"ParallelMap", makeShim<CmpMap, AstCmpMap>},
        {"SeriesMap", makeShim<CmpMap, AstCmpMap>},
        {"ZoomMap", makeShim<ZoomMap, AstZoomMap>},
    };
    assertOK();
    std::string const className(astGetC(rawObj, "Class"));
    auto name_caster = ClassCasterMap.find(className);
    if (name_caster == ClassCasterMap.end()) {
        throw std::runtime_error("Class " + className + " not supported");
    }
    auto ret = name_caster->second(rawObj);
    if (className == "CmpMap") {
        // cast to SeriesMap or ParallelMap as appropriate
        auto cmpMapPtr = std::static_pointer_cast<CmpMap>(ret);
        if (cmpMapPtr->getSeries()) {
            return std::static_pointer_cast<SeriesMap>(ret);
        } else {
            return std::static_pointer_cast<ParallelMap>(ret);
        }
    }
    return ret;
}

void Object::show(std::ostream & os) const {

    auto ch = astChannel(nullptr, sinkToOstream, "");

    // Store a poiner to the ostream in the channel, as required by sinkToOstream
    astPutChannelData(ch, &os);
    astWrite(ch, this->getRawPtr());
    astAnnul(ch);
    assertOK();
}

std::string Object::show() const {
    std::ostringstream os;
    show(os);
    return os.str();
}

}  // namespace ast
