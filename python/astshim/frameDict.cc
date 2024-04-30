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

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/set.h>

#include "lsst/cpputils/python.h"


namespace nb = nanobind;
using namespace nanobind::literals;

#include "astshim/Channel.h"
#include "astshim/Frame.h"
#include "astshim/FrameSet.h"
#include "astshim/FrameDict.h"
#include "astshim/Mapping.h"
#include "astshim/Stream.h"

namespace ast {
namespace {

/*
Make a FrameDict from a string output by Object.show()

Use this instead of the standard ObjectMaker for FrameDict
because ObjectMaker would return a FrameSet (since the serialization
is the same for both).
*/
class FrameDictMaker {
public:
    FrameDictMaker() = default;
    ~FrameDictMaker() = default;
    std::shared_ptr<Object> operator()(std::string const &state) {
        ast::StringStream stream(state);
        ast::Channel chan(stream);
        auto objPtr = chan.read();
        auto frameSetPtr = std::dynamic_pointer_cast<ast::FrameSet>(objPtr);
        if (!frameSetPtr) {
            throw std::runtime_error("Object being unpickled is a " + objPtr->getClassName() +
                                     " not a FrameSet");
        }
        return std::make_shared<ast::FrameDict>(*frameSetPtr);
    }
};
}

void wrapFrameDict(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyFrameDictMaker = nb::class_<FrameDictMaker>;
    static auto  clsMaker = wrappers.wrapType(PyFrameDictMaker(wrappers.module, "FrameDictMaker"), [](auto &mod, auto &cls) {
        cls.def(nb::init<>());
        cls.def("__call__", &FrameDictMaker::operator());
        cls.def("__reduce__",
                [cls](FrameDictMaker const &self) { return nb::make_tuple(cls, nb::tuple()); });
    });

    using PyFrameDict = nb::class_<FrameDict, FrameSet>;
    wrappers.wrapType(PyFrameDict(wrappers.module, "FrameDict"), [](auto &mod, auto &cls) {

        cls.def(nb::init<Frame const &, std::string const &>(), "frame"_a, "options"_a = "");
        cls.def(nb::init<Frame const &, Mapping const &, Frame const &, std::string const &>(), "baseFrame"_a,
                "mapping"_a, "currentFrame"_a, "options"_a = "");
        cls.def(nb::init<FrameSet const &>(), "frameSet"_a);
        cls.def(nb::init<FrameDict const &>());

        cls.def("copy", &FrameDict::copy);

        cls.def("addFrame", nb::overload_cast<int, Mapping const &, Frame const &>(&FrameDict::addFrame),
                "iframe"_a, "map"_a, "frame"_a);
        cls.def("addFrame",
                nb::overload_cast<std::string const &, Mapping const &, Frame const &>(&FrameDict::addFrame),
                "domain"_a, "map"_a, "frame"_a);
        cls.def("getAllDomains", &FrameDict::getAllDomains);
        cls.def("getFrame", nb::overload_cast<int, bool>(&FrameDict::getFrame, nb::const_), "index"_a,
                "copy"_a = true);
        cls.def("getFrame", nb::overload_cast<std::string const &, bool>(&FrameDict::getFrame, nb::const_),
                "domain"_a, "copy"_a = true);
        cls.def("getMapping", nb::overload_cast<int, int>(&FrameDict::getMapping, nb::const_),
                "from"_a = FrameDict::BASE, "to"_a = FrameDict::CURRENT);
        cls.def("getMapping", nb::overload_cast<int, std::string const &>(&FrameDict::getMapping, nb::const_),
                "from"_a = FrameDict::BASE, "to"_a = FrameDict::CURRENT);
        cls.def("getMapping", nb::overload_cast<std::string const &, int>(&FrameDict::getMapping, nb::const_),
                "from"_a = FrameDict::BASE, "to"_a = FrameDict::CURRENT);
        cls.def("getMapping",
                nb::overload_cast<std::string const &, std::string const &>(&FrameDict::getMapping, nb::const_),
                "from"_a = FrameDict::BASE, "to"_a = FrameDict::CURRENT);
        cls.def("getIndex", &FrameDict::getIndex, "domain"_a);
        cls.def("hasDomain", &FrameDict::hasDomain, "domain"_a);
        cls.def("mirrorVariants", nb::overload_cast<int>(&FrameDict::mirrorVariants), "index"_a);
        cls.def("mirrorVariants", nb::overload_cast<std::string const &>(&FrameDict::mirrorVariants), "domain"_a);
        cls.def("remapFrame", nb::overload_cast<int, Mapping &>(&FrameDict::remapFrame), "index"_a, "map"_a);
        cls.def("remapFrame", nb::overload_cast<std::string const &, Mapping &>(&FrameDict::remapFrame),
                "domain"_a, "map"_a);
        cls.def("removeFrame", nb::overload_cast<int>(&FrameDict::removeFrame), "index"_a);
        cls.def("removeFrame", nb::overload_cast<std::string const &>(&FrameDict::removeFrame), "domain"_a);
        cls.def("setBase", nb::overload_cast<int>(&FrameDict::setBase), "index"_a);
        cls.def("setBase", nb::overload_cast<std::string const &>(&FrameDict::setBase), "domain"_a);
        cls.def("setCurrent", nb::overload_cast<int>(&FrameDict::setCurrent), "index"_a);
        cls.def("setCurrent", nb::overload_cast<std::string const &>(&FrameDict::setCurrent), "domain"_a);
        cls.def("setDomain", &FrameDict::setDomain, "domain"_a);

        /// Override standard pickling support so we get a FrameDict back, instead of a FrameSet
        cls.def("__reduce__", [](Object const &self) {
            std::string state = self.show(false);
            auto unpickleArgs = nb::make_tuple(state);
            return nb::make_tuple(clsMaker(), unpickleArgs);
        });
    });
}

}  // namespace ast
