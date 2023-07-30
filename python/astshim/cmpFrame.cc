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
#include <memory>

#include <pybind11/pybind11.h>
#include "lsst/cpputils/python.h"

#include "astshim/CmpFrame.h"
#include "astshim/Frame.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ast {

void wrapCmpFrame(lsst::utils::python::WrapperCollection &wrappers) {
    using PyCmpFrame =  py::class_<CmpFrame, Frame>;
    wrappers.wrapType(PyCmpFrame(wrappers.module, "CmpFrame"), [](auto &mod, auto &cls) {
        cls.def(py::init<Frame const &, Frame const &, std::string const &>(), "frame1"_a, "frame2"_a,
                "options"_a = "");
        cls.def(py::init<CmpFrame const &>());
        cls.def("__getitem__", &CmpFrame::operator[], py::is_operator());
        cls.def("__len__", [](CmpFrame const &) { return 2; });
        cls.def("copy", &CmpFrame::copy);
    });
}

}  // namespace ast
