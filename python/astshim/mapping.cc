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
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "ndarray/nanobind.h"
#include "lsst/cpputils/python.h"

#include "astshim/base.h"
#include "astshim/Mapping.h"
#include "astshim/Object.h"
#include "astshim/ParallelMap.h"
#include "astshim/SeriesMap.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapMapping(lsst::cpputils::python::WrapperCollection &wrappers) {
using PyMapping = nb::class_<Mapping, Object>;
    wrappers.wrapType(PyMapping(wrappers.module, "Mapping"), [](auto &mod, auto &cls) {
        cls.def_prop_ro("nIn", &Mapping::getNIn);
        cls.def_prop_ro("nOut", &Mapping::getNOut);
        cls.def_prop_ro("isSimple", &Mapping::getIsSimple);
        cls.def_prop_ro("hasForward", &Mapping::hasForward);
        cls.def_prop_ro("hasInverse", &Mapping::hasInverse);
        cls.def_prop_ro("isInverted", &Mapping::isInverted);
        cls.def_prop_ro("isLinear", &Mapping::getIsLinear);
        cls.def_prop_rw("report", &Mapping::getReport, &Mapping::setReport);
        cls.def("copy", &Mapping::copy);
        cls.def("inverted", &Mapping::inverted);
        cls.def("linearApprox", &Mapping::linearApprox, "lbnd"_a, "ubnd"_a, "tol"_a);
        cls.def("then", &Mapping::then, "next"_a);
        cls.def("under", &Mapping::under, "next"_a);
        cls.def("rate", &Mapping::rate, "at"_a, "ax1"_a, "ax2"_a);
        cls.def("simplified", &Mapping::simplified);
        // wrap the overloads of applyForward, applyInverse, tranGridForward and tranGridInverse that return a new
        // result
        cls.def("applyForward", nb::overload_cast<ConstArray2D const &>(&Mapping::applyForward, nb::const_),
                "from"_a);
        cls.def("applyForward",
                nb::overload_cast<std::vector<double> const &>(&Mapping::applyForward, nb::const_), "from"_a);
        cls.def("applyInverse", nb::overload_cast<ConstArray2D const &>(&Mapping::applyInverse, nb::const_),
                "from"_a);
        cls.def("applyInverse",
                nb::overload_cast<std::vector<double> const &>(&Mapping::applyInverse, nb::const_), "from"_a);
        cls.def("tranGridForward",
                nb::overload_cast<PointI const &, PointI const &, double, int, int>(&Mapping::tranGridForward,
                                                                                    nb::const_),
                "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);
        cls.def("tranGridInverse",
                nb::overload_cast<PointI const &, PointI const &, double, int, int>(&Mapping::tranGridInverse,
                                                                                    nb::const_),
                "lbnd"_a, "ubnd"_a, "tol"_a, "maxpix"_a, "nPoints"_a);
    });
}

}  // namespace ast
