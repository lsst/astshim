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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "astshim/base.h"
#include "astshim/Mapping.h"
#include "astshim/Object.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace {

PYBIND11_PLUGIN(mapping) {
    py::module mod("mapping", "Python wrapper for Mapping");

    py::module::import("astshim.object");
    py::module::import("astshim.mapBox");
    py::module::import("astshim.mapSplit");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<Mapping, std::shared_ptr<Mapping>, Object> cls(mod, "Mapping");

    cls.def("copy", &Mapping::copy);
    cls.def("getNin", &Mapping::getNin);
    cls.def("getNout", &Mapping::getNout);
    cls.def("getIsSimple", &Mapping::getIsSimple);
    cls.def("isInverted", &Mapping::isInverted);
    cls.def("getIsLinear", &Mapping::getIsLinear);
    cls.def("getReport", &Mapping::getReport);
    cls.def("getTranForward", &Mapping::getTranForward);
    cls.def("getTranInverse", &Mapping::getTranInverse);
    cls.def("getInverse", &Mapping::getInverse);
    cls.def("linearApprox", &Mapping::linearApprox, "lbnd"_a, "ubnd"_a, "tol"_a);
    cls.def("of", &Mapping::of, "first"_a);
    cls.def("over", &Mapping::over, "first"_a);
    cls.def("rate", &Mapping::rate, "at"_a, "ax1"_a, "ax2"_a);
    cls.def("setReport", &Mapping::setReport, "report"_a);
    cls.def("simplify", &Mapping::simplify);
    // wrap the overloads of tranForward, tranInverse, tranGridForward and tranGridInverse that return a new
    // result
    cls.def("tranForward", (Array2D(Mapping::*)(ConstArray2D const &) const) & Mapping::tranForward,
            "from"_a);
    cls.def("tranForward",
            (std::vector<double>(Mapping::*)(std::vector<double> const &) const) & Mapping::tranForward,
            "from"_a);
    cls.def("tranInverse", (Array2D(Mapping::*)(ConstArray2D const &) const) & Mapping::tranInverse,
            "from"_a);
    cls.def("tranInverse",
            (std::vector<double>(Mapping::*)(std::vector<double> const &) const) & Mapping::tranInverse,
            "from"_a);
    cls.def("tranGridForward", (Array2D(Mapping::*)(PointI const &, PointI const &, double, int, int) const) &
                                       Mapping::tranGridForward,
            "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);
    cls.def("tranGridInverse", (Array2D(Mapping::*)(PointI const &, PointI const &, double, int, int) const) &
                                       Mapping::tranGridInverse,
            "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);

    return mod.ptr();
}

}  // <anonymous>
}  // ast
