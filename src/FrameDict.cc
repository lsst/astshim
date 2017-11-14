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
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "astshim/FrameDict.h"

namespace ast {

void FrameDict::addFrame(int iframe, Mapping const &map, Frame const &frame) {
    if (hasDomain(frame.getDomain())) {
        throw std::invalid_argument("A frame already exists with domain " + frame.getDomain());
    }
    auto copy = getFrameSet();
    copy->addFrame(iframe, map, frame);
    _update(*copy);
}

void FrameDict::addFrame(std::string const &domain, Mapping const &map, Frame const &frame) {
    addFrame(getIndex(domain), map, frame);
}

std::set<std::string> FrameDict::getAllDomains() const {
    std::set<std::string> domains;
    for (auto const &item : _domainIndexDict) {
        domains.emplace(item.first);
    }
    return domains;
}

void FrameDict::removeFrame(int iframe) {
    auto copy = getFrameSet();
    copy->removeFrame(iframe);
    _update(*copy);
}

void FrameDict::removeFrame(std::string const &domain) { removeFrame(getIndex(domain)); }

void FrameDict::setDomain(std::string const &domain) {
    if (getDomain() == domain) {
        // null rename
        return;
    }
    if (hasDomain(domain)) {
        throw std::invalid_argument("Another framea already has domain name " + domain);
    }
    auto copy = getFrameSet();
    copy->setDomain(domain);
    _update(*copy);
}

FrameDict::FrameDict(AstFrameSet *rawptr) : FrameSet(rawptr), _domainIndexDict() {
    _domainIndexDict = _makeNewDict(*this);
}

std::unordered_map<std::string, int> FrameDict::_makeNewDict(FrameSet const &frameSet) {
    std::unordered_map<std::string, int> dict;
    for (int index = 1, end = frameSet.getNFrame(); index <= end; ++index) {
        auto const domain = frameSet.getFrame(index, false)->getDomain();
        if (domain.empty()) {
            continue;
        } else if (dict.count(domain) > 0) {
            throw std::invalid_argument("More than one frame with domain " + domain);
        }
        dict[domain] = index;
    }
    return dict;
}

}  // namespace ast
