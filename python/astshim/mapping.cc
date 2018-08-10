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

PYBIND11_MODULE(mapping, mod) {
    py::module::import("astshim.object");
    py::module::import("astshim.mapBox");
    py::module::import("astshim.mapSplit");

    py::class_<Mapping, std::shared_ptr<Mapping>, Object> cls(mod, "Mapping");

    cls.def_property_readonly("nIn", &Mapping::getNIn);
    cls.def_property_readonly("nOut", &Mapping::getNOut);
    cls.def_property_readonly("isSimple", &Mapping::getIsSimple);
    cls.def_property_readonly("hasForward", &Mapping::hasForward);
    cls.def_property_readonly("hasInverse", &Mapping::hasInverse);
    cls.def_property_readonly("isInverted", &Mapping::isInverted);
    cls.def_property_readonly("isLinear", &Mapping::getIsLinear);
    cls.def_property("report", &Mapping::getReport, &Mapping::setReport);

    cls.def("copy", &Mapping::copy);
    cls.def("inverted", &Mapping::inverted);
    cls.def("linearApprox", &Mapping::linearApprox, "lbnd"_a, "ubnd"_a, "tol"_a);
    cls.def("then", &Mapping::then, "next"_a);
    cls.def("under", &Mapping::under, "next"_a);
    cls.def("rate", &Mapping::rate, "at"_a, "ax1"_a, "ax2"_a);
    cls.def("simplified", &Mapping::simplified);
    // wrap the overloads of applyForward, applyInverse, tranGridForward and tranGridInverse that return a new
    // result
    cls.def("applyForward", py::overload_cast<ConstArray2D const &>(&Mapping::applyForward, py::const_),
            "from"_a);
    cls.def("applyForward",
            py::overload_cast<std::vector<double> const &>(&Mapping::applyForward, py::const_), "from"_a);
    cls.def("applyInverse", py::overload_cast<ConstArray2D const &>(&Mapping::applyInverse, py::const_),
            "from"_a);
    cls.def("applyInverse",
            py::overload_cast<std::vector<double> const &>(&Mapping::applyInverse, py::const_), "from"_a);
    cls.def("tranGridForward",
            py::overload_cast<PointI const &, PointI const &, double, int, int>(&Mapping::tranGridForward,
                                                                                py::const_),
            "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);
    cls.def("tranGridInverse",
            py::overload_cast<PointI const &, PointI const &, double, int, int>(&Mapping::tranGridInverse,
                                                                                py::const_),
            "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);
}

}  // namespace
}  // namespace ast
