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
#ifndef ASTSHIM_TABLE_H
#define ASTSHIM_TABLE_H

#include <complex>
#include <string>
#include <vector>

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/KeyMap.h"

namespace ast {

class Table : public KeyMap {
    friend class Object;

public:
    explicit Table(std::string const &options = "")
            : KeyMap(reinterpret_cast<AstKeyMap *>(astTable("%s", options.c_str()))) {
        assertOK();;
    }

    virtual ~Table(){};

    Table(Table const &) = default;
    Table(Table &&) = default;
    Table &operator=(Table const &) = delete;
    Table &operator=(Table &&) = default;

    std::string columnName(int index) const {
        std::string name = astColumnName(getRawPtr(), index);
        assertOK();
        return name;
    }

    DataType columnType(std::string const &column) const {
        int retVal = Object::getI("ColumnType(" + column + ")");
        assertOK();
        return static_cast<DataType>(retVal);
    }

    int columnLength(std::string const &column) const {
        int retVal = Object::getI("ColumnLength(" + column + ")");
        assertOK();
        return retVal;
    }

    int columnLenC(std::string const &column) const {
        int retVal = Object::getI("ColumnLenC(" + column + ")");
        assertOK();
        return retVal;
    }

    int columnNdim(std::string const &column) const {
        int retVal = Object::getI("ColumnNdim(" + column + ")");
        assertOK();
        return retVal;
    }

    std::string columnUnit(std::string const &column) const {
        std::string retVal = Object::getC("ColumnUnit(" + column + ")");
        assertOK();
        return retVal;
    }

    std::vector<int> columnShape(std::string const &column) {
        int const mxdim = columnNdim(column);
        std::vector<int> dims(mxdim);
        if (mxdim > 0) {
            int ndim;
            astColumnShape(getRawPtr(), column.c_str(), mxdim, &ndim, dims.data());
        }
        assertOK();
        return dims;
    }

    /**
    Get @ref Table_NColumn "NColumn": The number of columns currently in the Table
    */
    int getNColumn() const { return Object::getI("NColumn"); }

    /**
    Get @ref Table_NRow "NRow": The number of rows currently in the Table
    */
    int getNRow() const { return Object::getI("NRow"); }

protected:

    /**
    Construct a Table from a raw AstTable
    */
    explicit Table(AstTable *rawTable) : KeyMap(reinterpret_cast<AstKeyMap *>(rawTable)) {
        if (!astIsATable(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a Table";
            throw std::invalid_argument(os.str());
        }
        assertOK();
    }

};

} // namespace ast

#endif
