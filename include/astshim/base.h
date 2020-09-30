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
#ifndef ASTSHIM_BASE_H
#define ASTSHIM_BASE_H

#include <string>
#include <stdexcept>
#include <vector>

#include "ndarray.h"

extern "C" {
#include "star/ast.h"
}

// Do not delete this or free functions and enums will not be documented
/// AST wrapper classes and functions.
namespace ast {

/**
2D array of const double; typically used for lists of const points
*/
using Array2D = ndarray::Array<double, 2, 2>;
/**
2D array of const double; typically used for lists of const points
*/
using ConstArray2D = ndarray::Array<const double, 2, 2>;
/**
Vector of ints; typically used for the bounds of Mapping.tranGridForward and inverse
*/
using PointI = std::vector<int>;
/**
Vector of double; used for bounds, points

Also used to store a list of points as sequential data,
as an alternative to Array2D and ConstArray2D
*/
using PointD = std::vector<double>;

/**
Data types held by a KeyMap
*/
enum class DataType {
    IntType = AST__INTTYPE,
    ShortIntType = AST__SINTTYPE,
    ByteType = AST__BYTETYPE,
    DoubleType = AST__DOUBLETYPE,
    FloatType = AST__FLOATTYPE,
    StringType = AST__STRINGTYPE,
    ObjectType = AST__OBJECTTYPE,
    PointerType = AST__POINTERTYPE,
    UndefinedType = AST__UNDEFTYPE,
    BadType = AST__BADTYPE
};

/**
Reshape a vector as a 2-dimensional array that shares the same memory

To convert a vector of coefficients to an array of coefficients
for PolyMap or ChebyMap, call this with nAxes = nPoints / width,
where width is the number of elements in each coefficient:
width = nOut + 2 for forward coefficients, nIn + 2 for inverse coefficients.

@param[in] vec  Vector of points, with all values for one axis first,
    then the next axes, and so on, e.g. x1, x2, ...xnPt, y1, y2, ...ynNpt
@param[in] nAxes  Number of axes per point
@return 2-dimensional array with dimensions (nPts, nAxes)
@throws std::runtime_error if vec length is not a multiple of nAxes

@warning You must hold onto the original vector until you are done
with the returned array, else the array will be corrupted.
(However, the Python version returns a copy, to avoid memory issues.)
@{
*/
ConstArray2D arrayFromVector(std::vector<double> const &vec, int nAxes);

Array2D arrayFromVector(std::vector<double> &vec, int nAxes);
/// @}

/**
Throw std::runtime_error if AST's state is bad

@param  rawPtr1  An AST object to free if status is bad
@param  rawPtr2  An AST object to free if status is bad

@note on the first call an error handler is registered
that saves error messages to a buffer.
*/
void assertOK(AstObject *rawPtr1 = nullptr, AstObject *rawPtr2 = nullptr);

/**
Control whether graphical escape sequences are included in strings.

The `Plot` class defines a set of escape sequences which can be
included within a text string in order to control the appearance of
sub-strings within the text. (See the Escape attribute for a
description of these escape sequences). It is usually inappropriate
for AST to return strings containing such escape sequences when
called by application code. For instance, an application which
displays the value of the @ref Frame_Title "Title" attribute
of a Frame usually does
not want the displayed string to include potentially long escape
sequences which a human read would have difficuly interpreting.
Therefore the default behaviour is for AST to strip out such escape
sequences when called by application code. This default behaviour
can be changed using this function.

@param[in] include  Possible values are:
                -  -1 (or any negative value) to return the current value without changing it.
                -   0 to not include escape sequences,
                -   1 (or any positive value) to include escape sequences,
@return the previous value (or current value if `include` is negative).

### Notes:

- This function also controls whether the AST function `astStripEscapes` removes escape sequences
    from the supplied string, or returns the supplied string without change.
- Unlike the AST function `astEscapes`, this function will not attempt to execute
    if an error has already occurred.
*/
inline bool escapes(int include = -1) {
    assertOK();
    int ret = astEscapes(include);
    assertOK();
    return ret;
}

inline int ast_version(void) {
  return astVersion;
}

}  // namespace ast

#endif
