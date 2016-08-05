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
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "astshim/base.h"
#include "astshim/Object.h"

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
