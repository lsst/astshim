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
#include "lsst/cpputils/python.h"

#include "astshim/detail/testUtils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {
namespace detail {
namespace {
void declareUtils(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.module.def("makeFrameDict", makeFrameDict);
}
} // namespace

PYBIND11_MODULE(testUtils, mod) {
    using lsst::cpputils::python::WrapperCollection;
    lsst::cpputils::python::WrapperCollection wrappers(mod, "astshim.detail.testUtils");
    declareUtils(wrappers);
    wrappers.finish();
}

}  // namespace detail
}  // namespace ast
