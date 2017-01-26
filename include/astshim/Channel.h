/* 
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#ifndef ASTSHIM_CHANNEL_H
#define ASTSHIM_CHANNEL_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Stream.h"

namespace ast {

/**
Channel provides input/output of AST objects.

Writing an @ref Object to a @ref Channel will generate a textual
representation of that @ref Object, and reading from a @ref Channel will
create a new @ref Object from its textual representation.

Note that a channel cannot be deep-copied because the contained stream cannot be deep-copied

### Missing Methods

- astWarnings has not been made a method because KeyMap is not yet wrapped.

- astPutChannelData is used internally and should not be called directly.

### Attributes

@ref Channel provides the following attributes, in addition to those provided by @ref Object

- @ref Channel_Comment "Comment": Include textual comments in output?
- @ref Channel_Full "Full": Set level of output detail.
- @ref Channel_Indent "Indent": Indentation increment between objects.
- @ref Channel_ReportLevel "ReportLevel": Selects the level of error reporting
- @ref Channel_Skip "Skip": Skip irrelevant data?
- @ref Channel_Strict "Strict": Generate errors instead of warnings?
*/
class Channel : public Object {
friend class Object;
public:
    /**
    Construct a channel that uses a provided @ref Stream

    @param[in] stream  Stream for channel I/O:
        - For file I/O: provide a @ref FileStream
        - For string I/O (e.g. unit tests): provide a @ref StringStream
        - For standard I/O provide `Stream(&std::cin, &std::cout))`
            where either stream can be nullptr if not wanted
    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit Channel(Stream & stream, std::string const & options="");

    virtual ~Channel();

    Channel(Channel const &) = delete;
    Channel(Channel &&) = default;
    Channel & operator=(Channel const &) = delete;
    Channel & operator=(Channel &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<Channel> copy() const {
        throw std::logic_error(
            "Channel cannot be deep copied because its contained stream cannot be deep copied");
    }

    /// Get @ref Channel_Comment "Comment": include textual comments in output?
    bool getComment() const { return getB("Comment"); }

    /// Get @ref Channel_Full "Full": level of output detail; one of -1: minimum, 0: normal, 1: verbose.
    int getFull() const { return getI("Full"); }

    /// Get @ref Channel_Indent "Indent": indentation increment between objects.
    int getIndent() const { return getB("Indent"); }

    /// Get @ref Channel_ReportLevel "ReportLevel": report level.
    int getReportLevel() const { return getI("ReportLevel"); }

    /// Get @ref Channel_Skip "Skip": skip irrelevant data on input?
    bool getSkip() const { return getB("Skip"); }

    /// Get @ref Channel_Strict "Strict": generate errors instead of warnings?
    bool getStrict() const { return getB("Strict"); }

    /// Read an object from a channel.
    std::shared_ptr<Object> read();

    /// Set @ref Channel_Comment "Comment": include textual comments in output?
    void setComment(bool skip) { setB("Comment", skip); }

    /// Set @ref Channel_Full "Full": level of output detail; one of -1: minimum, 0: normal, 1: verbose.
    void setFull(int full) { setI("Full", full); }

    /// Set @ref Channel_Indent "Indent": indentation increment between objects.
    void setIndent(int indent) { setI("Indent", indent); }

    /**
    Set @ref Channel_ReportLevel "ReportLevel": report level; an integer in the range [0, 3]
    where 0 is the most verbose.

    @throw std::invalid_argument if level is not in range [0, 3]
    */
    void setReportLevel(int level) {
        if ((level < 0) || (level > 3)) {
            std::ostringstream os;
            os << "level = " << level << " not in range [0, 3]";
            throw std::invalid_argument(os.str());
        }
        setI("ReportLevel", level);
    }

    /// Set @ref Channel_Skip "Skip": skip irrelevant data on input?
    void setSkip(bool skip) { setB("Skip", skip); }

    /// Set @ref Channel_Strict "Strict": generate errors instead of warnings?
    void setStrict(bool strict) { setB("Strict", strict); }

    /// Write an object to a channel.
    int write(Object const & object);

protected:
    virtual std::shared_ptr<Object> _copyPolymorphic() const {
        return std::shared_ptr<Object>();
    }    

    /**
    Construct a channel from an AST channel pointer and a @ref Stream

    This is the constructor most subclasses use for their high-level constructor.
    */
    explicit Channel(AstChannel * chan, Stream & stream);

    /**
    Construct a channel from an AST channel pointer that has its own stream
    */
    explicit Channel(AstChannel * chan);

    Stream _stream;
};

}  // namespace ast

#endif
