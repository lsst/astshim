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
/**
Demonstrate AST reference counting

This shows that AST handles reference counting nicely:
if an AST function returns an object, you own it (even if the object
you retrieved is from a collection such as AstCmpMap or AstFrameSet;
in that case you get a shallow copy).
*/
extern "C" {
  #include "ast.h"
}
#include <iostream>

int main() {
    AstZoomMap *zoomMap = astZoomMap(2, 5, "");
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;
    double shift[] = {1.0, -0.5};
    AstShiftMap *shiftMap = astShiftMap(2, shift, "");
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;

    std::cout << "\nAfter adding these to a CmpMap\n";
    AstCmpMap *cmpMap = astCmpMap(shiftMap, zoomMap, true, "");
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;

    std::cout << "\nAfter retrieving these from the CmpMap\n";
    AstMapping *map1;
    AstMapping *map2;
    int series, inverted1, inverted2;
    astDecompose(cmpMap, &map1, &map2, &series, &inverted1, &inverted2);
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;

    std::cout << "\nAfter deleting the CmpMap\n";
    astAnnul(cmpMap);
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;
    std::cout << "zoomMap refcount = " << astGetI(zoomMap, "RefCount") << std::endl;
}
