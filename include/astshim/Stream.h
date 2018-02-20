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
#ifndef ASTSHIM_SOURCESINK_H
#define ASTSHIM_SOURCESINK_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/Object.h"

namespace ast {

class Channel;  // forward declaration for friendship

/**
A stream for ast::Channel
*/
class Stream {
public:
    /**
    Construct a Stream from input and output std::streams

    @param[in] istreamPtr  source/input stream; may match `ostreamPtr`;
                        may be nullptr if sourcing not needed
    @param[in] ostreamPtr  sink/output stream; may match `istreamPtr`;
                        may be nullptr if sinking not needed
    */
    explicit Stream(std::istream *istreamPtr, std::ostream *ostreamPtr)
            : _istreamPtr(), _ostreamPtr(), _sourceStr(), _isFits(false) {
        if (istreamPtr) {
            _istreamPtr = std::make_shared<std::istream>(istreamPtr->rdbuf());
        }
        if (ostreamPtr) {
            _ostreamPtr = std::make_shared<std::ostream>(ostreamPtr->rdbuf());
        }
    }

    explicit Stream() : Stream(nullptr, nullptr) {}

    virtual ~Stream() {}

    Stream(Stream const &) = default;
    Stream(Stream &&) = default;
    Stream &operator=(Stream const &) = default;
    Stream &operator=(Stream &&) = default;

    /**
    Return true if this Stream has an input or output std::stream
    */
    bool hasStdStream() { return _istreamPtr || _ostreamPtr; }

    /**
    Source (read) from the stream

    @return  the data read as a C string,
        or nullptr if there is no source stream or the source stream is empty
        or in an error state.
        The Stream owns the string buffer, and it will be invalidated on the next
        call to this function.
    */
    char const *source() {
        if ((_istreamPtr) && (*_istreamPtr)) {
            if (_isFits) {
                // http://codereview.stackexchange.com/a/28759
                _sourceStr.resize(detail::FITSLEN);
                _istreamPtr->read(&_sourceStr[0], detail::FITSLEN);
            } else {
                std::getline(*_istreamPtr, _sourceStr);
            }
            if (*_istreamPtr) {
                return _sourceStr.c_str();
            }
        }
        return nullptr;
    }

    /**
    Sink (write) to the stream

    @param[in] cstr  data to write; a newline is then written if _isFits false
    @return true on success or if there is no stream pointer (a normal mode),
        false if the stream pointer is in a bad state after writing

    @note this function is not virtual because of type slicing: this function is called from code
    that casts a void pointer to a Stream pointer without knowing which kind of Stream it is.
    */
    bool sink(char const *cstr) {
        if (_ostreamPtr) {
            (*_ostreamPtr) << cstr;
            if (!_isFits) {
                (*_ostreamPtr) << std::endl;
            }
            return static_cast<bool>(*_ostreamPtr);
        } else {
            return true;
        }
    }

    friend class Channel;

    /// get isfits
    bool getIsFits() const { return _isFits; }

protected:
    /// set isFits
    void setIsFits(bool isFits) { _isFits = isFits; }

    std::shared_ptr<std::istream> _istreamPtr;  ///< input stream
    std::shared_ptr<std::ostream> _ostreamPtr;  ///< output stream
    /// string containing a local copy of sourced data,
    /// so @ref source can return a `char *` that won't disappear right away
    std::string _sourceStr;
    bool _isFits;  ///< is this a FITS stream?
};

/**
File-based source or sink (not both) for channels
*/
class FileStream : public Stream {
public:
    /**
    Construct a FileStream for reading or writing, but not both

    @param[in] path  Path to file as a string
    @param[in] doWrite  If true then write to the file, otherwise read from the file
    */
    explicit FileStream(std::string const &path, bool doWrite = false) : Stream(), _path(path) {
        auto mode = doWrite ? std::ios_base::out : std::ios_base::in;
        auto fstreamPtr = std::make_shared<std::fstream>(path, mode);
        if (!*fstreamPtr) {
            std::ostringstream os;
            os << "Failed to open file \"" << path << "\" for " << (doWrite ? "writing" : "reading");
            throw std::runtime_error(os.str());
        }
        if (doWrite) {
            _ostreamPtr = fstreamPtr;
        } else {
            _istreamPtr = fstreamPtr;
        }
    }

    virtual ~FileStream() {}

    /// Get the path to the file, as a string
    std::string getPath() const { return _path; }

private:
    std::string _path;  ///< Path to file
};

/**
String-based source and sink for channels

This sources from one stringstream and sinks to another.
The data can be retrieved at any time, without affecting the stream, using getData.
*/
class StringStream : public Stream {
public:
    /**
    Construct a StringStream

    @param[in] data  initial data for the source stream
    */
    explicit StringStream(std::string const &data = "") : Stream(), _istringstreamPtr(), _ostringstreamPtr() {
        _istringstreamPtr = std::make_shared<std::istringstream>(data);
        _ostringstreamPtr = std::make_shared<std::ostringstream>();
        _istreamPtr = _istringstreamPtr;
        _ostreamPtr = _ostringstreamPtr;
    }

    virtual ~StringStream() {}

    /// Get a copy of the text from the sink/output stream, without changing the stream
    std::string getSourceData() const { return _istringstreamPtr->str(); }

    /// Get a copy of the text from the sink/output stream, without changing the stream
    std::string getSinkData() const { return _ostringstreamPtr->str(); }

    /// Move output/sink data to input/source
    void sinkToSource() {
        _istringstreamPtr->clear();
        _istringstreamPtr->str(getSinkData());
        _ostringstreamPtr->str("");
    }

private:
    /// input stream as an istringstream, so stringstream-specific methods are available
    std::shared_ptr<std::istringstream> _istringstreamPtr;
    /// output stream as an ostringstream, so stringstream-specific methods are available
    std::shared_ptr<std::ostringstream> _ostringstreamPtr;
};

namespace detail {

/**
Source function that allows astChannel to source from a Stream

This function retrieves a pointer to a Stream `ssptr` using astChannelData,
then returns the result of calling `ssptr->source()`
*/
inline const char *source() {
    auto ssptr = reinterpret_cast<Stream *>(astChannelData);
    if (ssptr) {
        return ssptr->source();
    } else {
        return nullptr;
    }
}

/**
Sink function that allows astChannel to sink to a Stream

This function retrieves a pointer to a Stream `ssptr` using astChannelData,
then calls `ssptr->sink(cstr)`.
*/
inline void sink(const char *cstr) {
    auto ssptr = reinterpret_cast<Stream *>(astChannelData);
    if (ssptr) {
        auto isok = ssptr->sink(cstr);
        if (!isok) {
            astSetStatus(AST__ATGER);
        }
    }
}

}  // namespace detail

}  // namespace ast

#endif
