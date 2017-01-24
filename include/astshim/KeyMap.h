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
#ifndef ASTSHIM_KEYMAP_H
#define ASTSHIM_KEYMAP_H

#include <complex>

#include "astshim/base.h"
#include "astshim/Object.h"

namespace ast {
/**
KeyMap is used to store a set of values with associated keys which identify the values.

The keys are strings. These may be case sensitive or insensitive as selected by the KeyCase attribute,
and trailing spaces are ignored. The value associated with a key can be:
- integer (signed 4 and 2 byte, or unsigned 1 byte)
- floating point (single or double precision)
- void pointer
- character string
- AST raw object pointer

Each value can be a scalar or a one-dimensional vector.

### Attributes

KeyMap has the following attributes, in addition to those inherited from @ref Object

- `KeyCase`: Sets the case in which keys are stored
- `KeyError`: Report an error if the requested key does not exist?
- `SizeGuess`: The expected size of the KeyMap.
- `SortBy`: Determines how keys are sorted in a KeyMap.
- `MapLocked`: Prevent new entries being added to the KeyMap?
*/
class KeyMap : public Object {
public:
    /**
    Construct an empty KeyMap

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit KeyMap(std::string const & options="")
    :
        Object(reinterpret_cast<Object *>(astKeyMap(options.c_str())))
    {}

    virtual ~KeyMap();

    KeyMap(KeyMap const &) = delete;
    KeyMap(KeyMap &&) = default;
    KeyMap & operator=(KeyMap const &) = delete;
    KeyMap & operator=(KeyMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<KeyMap> copy() const {
        return std::static_pointer_cast<KeyMap>(_copyPolymorphic());
    }

    /**
    Get the value of a complex double card by key name

    @param[in] name  Name of keyword
    @param[in] defval  Value to return if keyword not found
    @return value as a @ref FoundValue, where found is false if the keyword was not found

    ### Notes

    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<std::complex<double>> getFitsCF(std::string const & name, std::complex<double> defval={0, 0});

    /**
    Get the value of a CONTINUE card by key name

    CONTINUE cards are treated like string values, but are encoded without an equals sign.

    @param[in] name  name of keyword
    @param[in] defval  value to return if keyword not found
    @return value as a FoundValue, where found is false if the keyword was not found

    ### Notes

    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<std::string> getFitsCN(std::string const & name, std::string defval="");

    /**
    Get the value of a double card by key name

    @param[in] name  name of keyword
    @param[in] defval  value to return if keyword not found
    @return value as a FoundValue, where found is false if the keyword was not found

    ### Notes

    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<double> getFitsF(std::string const & name, double defval=0);

    /**
    Get the value of a int card by key name

    @param[in] name  name of keyword
    @param[in] defval  value to return if keyword not found
    @return value as a FoundValue, where found is false if the keyword was not found

    ### Notes

    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<int> getFitsI(std::string const & name, int defval=0);

    /**
    Get the value of a bool card by key name

    @param[in] name  Name of keyword
    @param[in] defval  Value to return if keyword not found
    @return value as a FoundValue, where found is false if the keyword was not found

    ### Notes

    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<bool> getFitsL(std::string const & name, bool defval=false);

    /**
    Get the value of a string card by key name

    @param[in] name  Name of keyword
    @param[in] defval  Value to return if keyword not found
    @return value as a FoundValue, where found is false if the keyword was not found

    ### Notes

    - The FITS standard says that string keyword values should be padded with trailing spaces
    if they are shorter than 8 characters. For this reason, trailing spaces
    are removed from the returned string if the original string (including any trailing spaces)
    contains 8 or fewer characters. Trailing spaces are not removed from longer strings.
    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    FoundValue<std::string> getFitsS(std::string const & name, std::string defval="");

    /**
    Delete the current FITS card.

    The current card may be selected using the `Card` attribute (if its index is known)
    or by using `findFits` (if only the FITS keyword is known).
    After deletion, the following card becomes the current card.
    */
    void delFits() {
        astDelFits(getRawPtr());
        assertOK();
    }

    /**
    Delete all cards and associated information from a @ref KeyMap

    ### Notes

    - This method simply deletes the cards currently in the @ref KeyMap.
        Unlike astWriteFits, they are not first written out.
    - Any Tables or warnings stored in the @ref KeyMap are also deleted.
    */
    void emptyFits() {
        astEmptyFits(getRawPtr());
        assertOK();
    }

    /**
    Search for a card in a @ref KeyMap by keyword.

    The search commences at the current card (identified by the `Card`
    attribute) and ends when a card is found whose FITS keyword
    matches the template supplied, or when the last card in the
    @ref KeyMap has been searched.

    @warning this is very different than the `getFitsX` methods such as @ref getFitsS,
    whose search wraps around. In order to search all keys using @ref findFits you must
    first call @ref clearCard.

    If the search is successful (i.e. a card is found which matches the template),
    the contents of the card are returned and the `Card` attribute is adjusted
    to identify the card found (if `inc` false) or the one following it (if `inc` is true).

    If the search is not successful, the `Card` attribute is set to the "end-of-file".

    @param[in] name  The keyword to be found. In the simplest case,
                    this should simply be the keyword name (the search is case
                    insensitive and trailing spaces are ignored). However, this
                    template may also contain "field specifiers" which are
                    capable of matching a range of characters (see the "Keyword
                    Templates" section for details). In this case, the first card
                    with a keyword which matches the template will be found. To
                    find the next FITS card regardless of its keyword, you should
                    use the template "%f".
    @param[in] inc  If `false` (and the search is successful), the
                    @ref KeyMap's `Card` attribute will be set to the index of the card
                    that was found. If `true`, however, the `Card`
                    attribute will be incremented to identify the card which
                    follows the one found.

    @return data as a FoundValue

    ### Keyword Templates

    The templates used to match FITS keywords are normally composed
    of literal characters, which must match the keyword exactly
    (apart from case). However, a template may also contain "field
    specifiers" which can match a range of possible characters. This
    allows you to search for keywords that contain (for example)
    numbers, where the digits comprising the number are not known in
    advance.

    A field specifier starts with a "%" character. This is followed
    by an optional single digit (0 to 9) specifying a field
    width. Finally, there is a single character which specifies the

    type of character to be matched, as follows:

    - "c": matches all upper case letters,
    - "d": matches all decimal digits,
    - "f": matches all characters which are permitted within a FITS
        keyword (upper case letters, digits, underscores and hyphens).

    If the field width is omitted, the field specifier matches one
    or more characters. If the field width is zero, it matches zero
    or more characters. Otherwise, it matches exactly the number of
    characters specified. In addition to this:

    - The template "%f" will match a blank FITS keyword consisting
        of 8 spaces (as well as matching all other keywords).
    - A template consisting of 8 spaces will match a blank keyword (only).

    For example:

    - The template "BitPix" will match the keyword "BITPIX" only.
    - The template "crpix%1d" will match keywords consisting of "CRPIX" followed by one decimal digit.
    - The template "P%c" will match any keyword starting with "P" and followed by one or more letters.
    - The template "E%0f" will match any keyword beginning with "E".
    - The template "%f" will match any keyword at all (including a blank one).

    ### Examples

    `auto foundvalue = fitschan.findFits("%f", true)`

        Return the current FITS card and advance the `Card` attribute
        to the card that follows (the "%f" template matches any keyword).

    `auto foundvalue = fitschan.findFits("BITPIX", false)

        Return the next FITS card with the "BITPIX" keyword
        and leave the `Card` attribute pointing to it.
        You might wish to then call `setFitsI(...)` to modify its value.

    `auto foundvalue = fitscan.findFits("COMMENT", true)`

        Return the next COMMENT card and advance the `Card` attribute
        to the card that follows.

    `auto foundvalue = fitschan.findFits("CRVAL%1d", true)`

        Return the next keyword of the form "CRVALi" (for example,
        any of the keywords "CRVAL1", "CRVAL2" or "CRVAL3" would be matched).
        Advance the `Card` attribute to the card that follows.
    */
    FoundValue<std::string> findFits(std::string const & name, bool inc);

    /**
    Delete all cards in a @ref KeyMap that relate to any of the recognised WCS encodings.

    On exit, the current card is the first remaining card in the @ref KeyMap.
    */
    void purgeWcs() {
        astPurgeWCS(getRawPtr());
        assertOK();
    }

    /**
    Replace all FITS header cards.

    The cards are supplied concatenated together into a single character string.
    Any existing cards in the @ref KeyMap are removed before the new cards
    are added. The @ref KeyMap is "re-wound" on exit.
    This means that a subsequent invocation of read can be made immediately
    without the need to re-wind the @ref KeyMap first.

    @param[in] cards  A string containing the FITS cards to be stored.
                    Each individual card should occupy 80 characters in this string,
                    and there should be no delimiters, new lines, etc, between adjacent cards.
                    The final card may be less than 80 characters long.
                    This is the format produced by the fits_hdr2str function in the
                    CFITSIO library.
    */
    void putCards(std::string const & cards) {
        astPutCards(getRawPtr(), cards.c_str());
        assertOK();
    }

    /**
    Store a FITS header card in a @ref KeyMap.

    The card is either inserted before the current card
    or over-writes the current card, depending on `overwrite`.

    @param[in] card
    @param[in] overwrite   if `false`, the new card is inserted in before the current card.
                If `true` the new card replaces the current card.  In either case, the `Card`
                attribute is then incremented by one so that it subsequently identifies the card
                following the one stored.

    ### Notes

    - If the `Card` attribute initially points at the " end-of-file"
    (i.e. exceeds the number of cards in the @ref KeyMap), then the new card
    is appended as the last card in the @ref KeyMap.
    - An error will result if the supplied string cannot be interpreted as a FITS header card.
    */
    void putFits(std::string const & card, bool overwrite) {
        astPutFits(getRawPtr(), card.c_str(), overwrite);
        assertOK();
    }

    /**
    Read cards from the source and store them in the @ref KeyMap.

    This normally happens once-only, when the @ref KeyMap is accessed for the first time.
    This function provides a means of forcing a re-read of the external source,
    and may be useful if (say) new cards have been deposited into the external source.
    Any new cards read from the source are appended to the end of the current contents of the @ref KeyMap.

    ### Notes
  This is a no-op if the @ref Stream has no source.
    */
    void readFits() {
        astReadFits(getRawPtr());
        assertOK();
    }

    /**
    Keep the current card when an @ref Object is read that uses the card.

    Cards that have not been flagged in this way are removed when a read operation completes succesfully,
    but only if the card was used in the process of creating the returned @ref Object.
    Any cards that are irrelevant to the creation of the @ref Object are retained
    whether or not they are flagged.
    */
    void retainFits() {
        astRetainFits(getRawPtr());
        assertOK();
    }

    /**
    Create a new card of type std::complex<double>, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.
    */
    void setFitsCF(std::string const & name, std::complex<double> value,
                   std::string const & comment="", bool overwrite=false) {
        // this use of reinterpret_cast is explicitly permitted, for C compatibility
        astSetFitsCF(getRawPtr(), name.c_str(), reinterpret_cast<double(&)[2]>(value),
                     comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new comment card, possibly overwriting the current card

    A comment card is a card with no keyword name and no equals sign

    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsCM(std::string const & comment, bool overwrite=false) {
        astSetFitsCM(getRawPtr(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new "CONTINUE" card, possibly overwriting the current card

    "CONTINUE" cards are treated like string values, but are encoded without an equals sign.

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsCN(std::string const & name, std::string value,
                   std::string const & comment="", bool overwrite=false) {
        astSetFitsCN(getRawPtr(), name.c_str(), value.c_str(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card of type double, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsF(std::string const & name, double value,
                  std::string const & comment="", bool overwrite=false) {
        astSetFitsF(getRawPtr(), name.c_str(), value, comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card of type int, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsI(std::string const & name, int value,
                  std::string const & comment="", bool overwrite=false) {
        astSetFitsI(getRawPtr(), name.c_str(), value, comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card of type bool, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsL(std::string const & name, bool value,
                  std::string const & comment="", bool overwrite=false) {
        astSetFitsL(getRawPtr(), name.c_str(), value, comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card of type string, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] value  the value of the card.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - The FITS standard says that string keyword values should be padded with trailing spaces
    if they are shorter than 8 characters. For this reason, trailing spaces
    are removed from the returned string if the original string (including any trailing spaces)
    contains 8 or fewer characters. Trailing spaces are not removed from longer strings.
    - If the requested card is found, it becomes the current card;
    otherwise the current card is left pointing at the end-of-file
    - If the stored keyword values is not of the requested type, it is converted
    (if possible) into the requested type
    - If the keyword is found but has no associated value, an error is reported.
    If necessary, the testFits function can be used to determine if the keyword
    has a defined value, prior to calling this function.
    */
    void setFitsS(std::string const & name, std::string value,
                  std::string const & comment="", bool overwrite=false) {
        astSetFitsS(getRawPtr(), name.c_str(), value.c_str(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card with an undefined value, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the " name" parameter is used instead. If "name"
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the `Card` attribute).  If `false`, the new card is inserted
            before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - To assign a new value for an existing keyword, first find the card describing the keyword
        using `findFits`, and then use the appropriate `setFits` function to over-write the old value.
    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsU(std::string const & name,
                  std::string const & comment="", bool overwrite=false) {
        astSetFitsU(getRawPtr(), name.c_str(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Write all the cards in the channel to standard output
    */
    void showFits() const {
        astShowFits(getRawPtr());
        assertOK();
    }

    /**
    Determine if a named keyword is present, and if so, whether it has a value.

    ### Notes

    - This function does not change the current card.
    - The card following the current card is checked first. If this is not the required card,
      then the rest of the @ref KeyMap is searched, starting with the first card added to the @ref KeyMap.
      Therefore cards should be accessed in the order they are stored in the @ref KeyMap (if possible)
      as this will minimise the time spent searching for cards.
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    FitsKeyState testFits(std::string const & name) const;

    /**
    Write out all cards currently in the channel and clear the channel.
    */
    void writeFits() {
        astWriteFits(getRawPtr());
        assertOK();
    }

    /// Rewind the card index to the beginning
    void clearCard() { clear("Card"); }

    /// Get the index of the current card, where 1 is the first card
    int getCard() { return getI("Card"); }

    /// Set the index of the current card, where 1 is the first card
    void setCard(int ind) { setI("Card", ind); }

    /// Get the comment of the current card
    std::string getCardComm() const { return getC("CardComm"); }

    /// Get the keyword name of the current card
    std::string getCardName() const { return getC("CardName"); }

    /// Get the number of cards
    int getNcard() const { return getI("Ncard"); }

    /// Get the number of *unique* keywords
    int getNKey() const { return getI("NKey"); }

protected:
    virtual std::shared_ptr<Object> _copyPolymorphic() const {
        return _copyImpl<KeyMap, AstKeyMap>();
    }    

    /**
    Construct a KeyMap from a raw AstKeyMap
    */
    explicit KeyMap(AstKeyMap * rawKeyMap)
    :
        Object(reinterpret_cast<Object *>(rawKeyMap))
    {
        if (!astIsAFrameSet(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClass() << ", which is not a KeyMap";
            throw std::invalid_argument(os.str());
        }
    }

};

}  // namespace ast

#endif
