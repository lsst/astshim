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
#ifndef ASTSHIM_FITSTABLE_H
#define ASTSHIM_FITSTABLE_H

#include <complex>
#include <string>
#include <vector>

#include "ndarray.h"

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Table.h"
#include "astshim/FitsChan.h"

namespace ast {

class FitsTable : public Table {
    friend class Object;

public:
    explicit FitsTable(FitsChan const &header, std::string const &options = "")
            : Table(reinterpret_cast<AstTable *>(astFitsTable(const_cast<AstObject *>(header.getRawPtr()),
                                                              "%s", options.c_str()))) {
        assertOK();;
    }
    explicit FitsTable(std::string const &options = "")
            : Table(reinterpret_cast<AstTable *>(astFitsTable(NULL,
                                                              "%s", options.c_str()))) {
        assertOK();;
    }


    virtual ~FitsTable(){};

    FitsTable(FitsTable const &) = default;
    FitsTable(FitsTable &&) = default;
    FitsTable &operator=(FitsTable const &) = delete;
    FitsTable &operator=(FitsTable &&) = default;

    std::shared_ptr<FitsChan> getTableHeader() const {
        auto *rawFitsChan = reinterpret_cast<AstObject *>(astGetTableHeader(getRawPtr()));
        assertOK(rawFitsChan);
        if (!rawFitsChan) {
            throw std::runtime_error("getTableHeader failed (returned a null fitschan)");
        }
        return Object::fromAstObject<FitsChan>(rawFitsChan, false);
    }

    std::size_t columnSize(std::string const &column) {
        size_t retVal = astColumnSize(getRawPtr(), column.c_str());
        assertOK();
        return retVal;
    }

    // We do not know the shape of the column so in C++ we can only return
    // the elements as a 1-D double array. It is up to the caller to extract
    // the relevant information.
    ndarray::Array<double, 1, 1> getColumnData1D(std::string const &column) {
        auto dtype = columnType(column);
        if (dtype != DataType::DoubleType) {
            throw std::runtime_error("Data type not supported by getColumnData");
        }
        // We can ask AST for the number of bytes required but for now
        // calculate the number of elements from the shape and rows
        auto totnel = getNRow();
        auto shape = columnShape(column);
        for (auto &val : shape) {
            totnel *= val;
        }
        ndarray::Array<double, 1, 1> coldata = ndarray::allocate(ndarray::makeVector(totnel));
        int nelem;
        astGetColumnData(getRawPtr(), column.c_str(), AST__NANF, AST__NAN, totnel*sizeof(double),
                         coldata.getData(), &nelem);
        return coldata;
    }

  protected:

      /**
      Construct a FitsTable from a raw AstFitsTable
      */
      explicit FitsTable(AstFitsTable *rawFitsTable) : Table(reinterpret_cast<AstTable *>(rawFitsTable)) {
          if (!astIsAFitsTable(getRawPtr())) {
              std::ostringstream os;
              os << "this is a " << getClassName() << ", which is not a FitsTable";
              throw std::invalid_argument(os.str());
          }
          assertOK();
      }


};

} // namespace ast

#endif
