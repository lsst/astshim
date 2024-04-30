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
#include <string>

#include <nanobind/nanobind.h>

#include "lsst/cpputils/python.h"

#include "astshim/Frame.h"
#include "astshim/TimeFrame.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace ast {

void wrapTimeFrame(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyTimeFrame=nb::class_<TimeFrame, Frame>;
    wrappers.wrapType(PyTimeFrame(wrappers.module, "TimeFrame"), [](auto &mod, auto &cls) {

        cls.def(nb::init<std::string const &>(), "options"_a = "");
        cls.def(nb::init<TimeFrame const &>());

        cls.def_prop_rw("alignTimeScale", &TimeFrame::getAlignTimeScale, &TimeFrame::setAlignTimeScale);
        cls.def_prop_rw("ltOffset", &TimeFrame::getLTOffset, &TimeFrame::setLTOffset);
        cls.def_prop_rw("timeOrigin", &TimeFrame::getTimeOrigin, &TimeFrame::setTimeOrigin);
        cls.def_prop_rw("timeScale", &TimeFrame::getTimeScale, &TimeFrame::setTimeScale);

        cls.def("copy", &TimeFrame::copy);
        cls.def("currentTime", &TimeFrame::currentTime);
    });
}

}  // namespace ast
