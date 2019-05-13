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
#ifndef ASTSHIM_XMLCHAN_H
#define ASTSHIM_XMLCHAN_H

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Stream.h"
#include "astshim/Channel.h"

namespace ast {

/**
XmlChan provides input/output of AST objects.

Writing an Object to a XmlChan will generate a textual
representation of that Object, and reading from a XmlChan will
create a new Object from its textual representation.

@ref XmlChan provides the following attributes, in addition to those provided
by @ref Channel and @ref Object

@ref XmlChan_XmlFormat "XmlFormat": system for formatting Objects as XML.
@ref XmlChan_XmlLength "XmlLength": controls output buffer length; 0 for no limit.
@ref XmlChan_XmlPrefix "XmlPrefix": the namespace prefix to use when writing.
*/
class XmlChan : public Channel {
public:
    /**
    Construct a channel that uses a provided Stream

    @param[in] stream  Stream for channel I/O:
        - For file I/O: provide a FileStream
        - For string I/O (e.g. unit tests): provide a StringStream
        - For standard I/O provide `Stream(&std::cin, &std::cout))`
            where either stream can be nullptr if not wanted
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit XmlChan(Stream &stream, std::string const &options = "")
            : Channel(reinterpret_cast<AstChannel *>(
                              astXmlChan(detail::source, detail::sink, "%s", options.c_str())),
                      stream) {
        assertOK();
    }

    virtual ~XmlChan() {}

    XmlChan(XmlChan const &) = delete;
    XmlChan(XmlChan &&) = default;
    XmlChan &operator=(XmlChan const &) = delete;
    XmlChan &operator=(XmlChan &&) = default;

    /// Get @ref XmlChan_XmlFormat "XmlFormat" System for formatting Objects as XML.
    std::string getXmlFormat() const { return getC("XmlFormat"); }

    /// Get @ref XmlChan_XmlLength "XmlLength": controls output buffer length.
    int getXmlLength() const { return getI("XmlLength"); }

    /// Get @ref XmlChan_XmlPrefix "XmlPrefix": the namespace prefix to use when writing.
    std::string getXmlPrefix() { return getC("XmlPrefix"); }

    /// Set @ref XmlChan_XmlFormat "XmlFormat" System for formatting Objects as XML.
    void setXmlFormat(std::string const &format) { setC("XmlFormat", format); }

    /// Set @ref XmlChan_XmlLength "XmlLength": controls output buffer length; 0 for no limit.
    void setXmlLength(int len) { setI("XmlLength", len); }

    /// Set @ref XmlChan_XmlPrefix "XmlPrefix": the namespace prefix to use when writing.
    void setXmlPrefix(std::string const &prefix) { setC("XmlPrefix", prefix); }
};

}  // namespace ast

#endif
