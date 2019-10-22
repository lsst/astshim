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
#ifndef ASTSHIM_FITSCHAN_H
#define ASTSHIM_FITSCHAN_H

#include <complex>
#include <string>
#include <vector>

#include "astshim/base.h"
#include "astshim/Object.h"
#include "astshim/Stream.h"
#include "astshim/Channel.h"
#include "astshim/KeyMap.h"

namespace ast {

/**
Enums describing the presence or absence of a FITS keyword
*/
enum class FitsKeyState {
    ABSENT = 0,  ///< keyword is not present
    NOVALUE,     ///< keyword is present, but has no value
    PRESENT      ///< keyword is present and has a value
};

/**
Enums describing the FITS card type
*/
enum class CardType {
    NOTYPE = AST__NOTYPE,      ///< card does not exist (card number invalid)
    COMMENT = AST__COMMENT,    ///< card is a comment-style card with no "=" (COMMENT, HISTORY, ...)
    INT = AST__INT,            ///< integer
    FLOAT = AST__FLOAT,        ///< float
    STRING = AST__STRING,      ///< string
    COMPLEXF = AST__COMPLEXF,  ///< complex floating point
    COMPLEXI = AST__COMPLEXI,  ///< complex integer
    LOGICAL = AST__LOGICAL,    ///< boolean
    CONTINUE = AST__CONTINUE,  ///< CONTINUE card
    UNDEF = AST__UNDEF,        ///< card has no value
};

/**
A value and associated validity flag

One could use std::pair instead, but this is a bit nicer,
and also easier to python-wrap for complicated types.
*/
template <typename T>
class FoundValue {
public:
    /**
    Construct a FoundValue

    @param[in] found  Was the value found?
    @param[in] value  The value (must be a valid value, even if found false)
    */
    FoundValue(bool found, T const &value) : found(found), value(value) {}

    /// Default constructor: found false, value is default-constructed
    FoundValue() : found(false), value() {}
    bool found;  ///< Was the value found?
    T value;     ///< The found value; ignore if `found` is false
};

/**
A specialized form of \ref Channel which reads and writes FITS header cards

Writing an @ref Object to a @ref FitsChan will, if the @ref Object is suitable, generate a
description of that @ref Object composed of FITS header cards, and
reading from a @ref FitsChan will create a new @ref Object from its FITS
header card description.

While a @ref FitsChan is active, it represents a buffer which may
contain zero or more 80-character "header cards" conforming to
FITS conventions. Any sequence of FITS-conforming header cards
may be stored, apart from the "END" card whose existence is
merely implied.  The cards may be accessed in any order by using
the @ref FitsChan's @ref FitsChan_Card "Card" attribute, which identifies a "current"
card, to which subsequent operations apply. Searches
based on keyword may be performed (using @ref findFits), new
cards may be inserted (@ref putFits, @ref putCards, @ref setFitsS and similar)
and existing ones may be deleted with @ref delFits, extracted with @ref getFitsS and similar,
or changed with @ref setFitsS and similar.

### Missing Methods

Tables are not yet supported, so the following AST functions are not wrapped as methods:
- astGetTables
- astPutTable
- astPutTables
- astRemoveTables
- astTableSource

### Attributes

@ref FitsChan has the following attributes, in addition to those
provided by @ref Channel and @ref Object

- @ref FitsChan_AllWarnings "AllWarnings": A list of the available conditions
- @ref FitsChan_Card "Card": Index of current FITS card in a FitsChan
- @ref FitsChan_CardComm "CardComm": The comment of the current FITS card in a FitsChan
- @ref FitsChan_CardName "CardName": The keyword name of the current FITS card in a FitsChan
- @ref FitsChan_CardType "CardType": The data type of the current FITS card in a FitsChan
- @ref FitsChan_CarLin "CarLin": Ignore spherical rotations on CAR projections?
- @ref FitsChan_CDMatrix "CDMatrix": Use a CD matrix instead of a PC matrix?
- @ref FitsChan_Clean "Clean": Remove cards used whilst reading even if an error occurs?
- @ref FitsChan_DefB1950 "DefB1950": Use FK4 B1950 as default equatorial coordinates?
- @ref FitsChan_Encoding "Encoding": System for encoding Objects as FITS headers
- @ref FitsChan_FitsAxisOrder "FitsAxisOrder": Sets the order of WCS axes within new FITS-WCS headers
- @ref FitsChan_FitsDigits "FitsDigits": Digits of precision for floating-point FITS values
- @ref FitsChan_FitsTol "FitsTol": Tolerance used for writing a FrameSet using a foreign encoding
- @ref FitsChan_Iwc "Iwc": Add a Frame describing Intermediate World Coords?
- @ref FitsChan_NCard "NCard": Number of FITS header cards in a FitsChan
- @ref FitsChan_Nkey "Nkey": Number of unique keywords in a FitsChan
- @ref FitsChan_SipOK "SipOK": Use Spitzer Space Telescope keywords to define distortion?
- @ref FitsChan_SipReplace "SipReplace": Ignore inverse SIP coefficients (replacing them with
        fit coefficients or an iterative inverse)?
- @ref FitsChan_TabOK "TabOK": Should the FITS "-TAB" algorithm be recognised?
- @ref FitsChan_PolyTan "PolyTan": Use PVi_m keywords to define distorted TAN projection?
- @ref FitsChan_Warnings "Warnings": Produces warnings about selected conditions

### Notes

- Call @ref setFitsU to store a keyword that has no associated value (i.e. a card with unknown value).
- Call @ref setFitsCM to store a pure comment card (i.e. a card with a blank keyword).
- To assign a new value for an existing keyword, first find the card describing the keyword
    using \ref findFits, and then use the appropriate `setFits` function (e.g. @ref setFitsS)
    to overwrite the old value.

As for any Channel, when you create a @ref FitsChan, you specify a
@ref Stream which sources and sinks data
by reading and writing FITS header cards. If you provide
a source, it is used to fill the @ref FitsChan with header cards
when it is accessed for the first time. If you do not provide a
source, the @ref FitsChan remains empty until you explicitly enter
data into it (e.g. using `putFits`, `putCards`, @ref write
or by using the SourceFile attribute to specifying a text file from
which headers should be read). When the @ref FitsChan is deleted, any
remaining header cards in the @ref FitsChan will be written to the sink.

Coordinate system information may be described using FITS header
cards using several different conventions, termed
"encodings". When an AST @ref Object is written to (or read from) a
@ref FitsChan, the value of the @ref FitsChan's Encoding attribute
determines how the @ref Object is converted to (or from) a
description involving FITS header cards. In general, different
encodings will result in different sets of header cards to
describe the same @ref Object. Examples of encodings include the DSS
encoding (based on conventions used by the STScI Digitised Sky
Survey data), the FITS-WCS encoding (based on a proposed FITS
standard) and the NATIVE encoding (a near loss-less way of
storing AST Objects in FITS headers).

The available encodings differ in the range of Objects they can
represent, in the number of @ref Object descriptions that can coexist
in the same @ref FitsChan, and in their accessibility to other
(external) astronomy applications (see the Encoding attribute
for details). Encodings are not necessarily mutually exclusive
and it may sometimes be possible to describe the same @ref Object in
several ways within a particular set of FITS header cards by
using several different encodings.

The detailed behaviour of @ref read and @ref write, when used with
a @ref FitsChan, depends on the encoding in use. In general, however,
all successful use of @ref read is destructive, so that FITS header cards
are consumed in the process of reading an @ref Object, and are
removed from the @ref FitsChan (this deletion can be prevented for
specific cards by calling the @ref retainFits function).
An unsuccessful call of @ref read
(for instance, caused by the @ref FitsChan not containing the necessary
FITS headers cards needed to create an @ref Object) results in the
contents of the @ref FitsChan being left unchanged.

If the encoding in use allows only a single @ref Object description
to be stored in a @ref FitsChan (e.g. the `DSS`, `FITS-WCS` and `FITS-IRAF`
encodings), then write operations using @ref write will
over-write any existing @ref Object description using that
encoding. Otherwise (e.g. the `NATIVE` encoding), multiple @ref Object
descriptions are written sequentially and may later be read
back in the same sequence.
*/
class FitsChan : public Channel {
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
    explicit FitsChan(Stream &stream, std::string const &options = "");

    virtual ~FitsChan();

    FitsChan(FitsChan const &) = delete;
    FitsChan(FitsChan &&) = default;
    FitsChan &operator=(FitsChan const &) = delete;
    FitsChan &operator=(FitsChan &&) = default;

    /**
    Delete the current FITS card.

    The current card may be selected using the @ref FitsChan_Card "Card" attribute (if its index is known)
    or by using `findFits` (if only the FITS keyword is known).
    After deletion, the following card becomes the current card.
    */
    void delFits() {
        astDelFits(getRawPtr());
        assertOK();
    }

    /**
    Delete all cards and associated information from a @ref FitsChan

    ### Notes

    - This method simply deletes the cards currently in the @ref FitsChan.
        Unlike astWriteFits, they are not first written out.
    - Any Tables or warnings stored in the @ref FitsChan are also deleted.
    */
    void emptyFits() {
        astEmptyFits(getRawPtr());
        assertOK();
    }

    /**
    Search for a card in a @ref FitsChan by keyword.

    The search commences at the current card (identified by the `Card`
    attribute) and ends when a card is found whose FITS keyword
    matches the template supplied, or when the last card in the
    @ref FitsChan has been searched.

    @warning this is very different than the `getFitsX` methods such as @ref getFitsS,
    whose search wraps around. In order to search all keys using @ref findFits you must
    first call @ref clearCard.

    If the search is successful (i.e. a card is found which matches the template),
    the contents of the card are returned and the @ref FitsChan_Card "Card" attribute is adjusted
    to identify the card found (if `inc` false) or the one following it (if `inc` is true).

    If the search is not successful, the @ref FitsChan_Card "Card" attribute is set to the "end-of-file".

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
                    @ref FitsChan's @ref FitsChan_Card "Card" attribute will be set to the index of the card
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

        Return the current FITS card and advance the @ref FitsChan_Card "Card" attribute
        to the card that follows (the "%f" template matches any keyword).

    `auto foundvalue = fitschan.findFits("BITPIX", false)

        Return the next FITS card with the "BITPIX" keyword
        and leave the @ref FitsChan_Card "Card" attribute pointing to it.
        You might wish to then call `setFitsI(...)` to modify its value.

    `auto foundvalue = fitscan.findFits("COMMENT", true)`

        Return the next COMMENT card and advance the @ref FitsChan_Card "Card" attribute
        to the card that follows.

    `auto foundvalue = fitschan.findFits("CRVAL%1d", true)`

        Return the next keyword of the form "CRVALi" (for example,
        any of the keywords "CRVAL1", "CRVAL2" or "CRVAL3" would be matched).
        Advance the @ref FitsChan_Card "Card" attribute to the card that follows.
    */
    FoundValue<std::string> findFits(std::string const &name, bool inc);

    /**
    Get the value of a complex double card

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<std::complex<double>> getFitsCF(std::string const &name = "",
                                               std::complex<double> defval = {0, 0}) const;

    /**
    Get the value of a CONTINUE card

    CONTINUE cards are treated like string values, but are encoded without an equals sign.

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<std::string> getFitsCN(std::string const &name = "", std::string defval = "") const;

    /**
    Get the value of a double card

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<double> getFitsF(std::string const &name = "", double defval = 0) const;

    /**
    Get the value of a int card

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<int> getFitsI(std::string const &name = "", int defval = 0) const;

    /**
    Get the value of a bool card

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<bool> getFitsL(std::string const &name = "", bool defval = false) const;

    /**
    Get the value of a string card

    @param[in] name  Name of keyword, or empty for the current card
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
    FoundValue<std::string> getFitsS(std::string const &name = "", std::string defval = "") const;

    /**
    Get the name of all cards, in order, starting from the first card

    Not "const" because the code changes the index of the current card while operating
    (but restores the original index when done).
    */
    std::vector<std::string> getAllCardNames();

    /**
    Get @ref FitsChan_AllWarnings "AllWarnings": a space separated list of
    all the conditions names recognized by the @ref FitsChan_Warnings "Warnings" attribute.
    */
    std::string getAllWarnings() const { return getC("AllWarnings"); }

    /**
    Get @ref FitsChan_Card "Card": the index of the current card, where 1 is the first card.
    */
    int getCard() const { return getI("Card"); }

    /**
    Get @ref FitsChan_CardComm "CardComm": the comment of the current card
    */
    std::string getCardComm() const { return getC("CardComm"); }

    /**
    Get @ref FitsChan_CardName "CardName": the keyword name of the current card
    */
    std::string getCardName() const { return getC("CardName"); }

    /**
    Get @ref FitsChan_CardType "CardType": data type of the current FITS card
    */
    CardType getCardType() const { return static_cast<CardType>(getI("CardType")); }

    /**
    Get @ref FitsChan_CarLin "CarLin": ignore spherical rotations on CAR projections?
    */
    bool getCarLin() const { return getB("CarLin"); }

    /**
    Get @ref FitsChan_CDMatrix "CDMatrix": use CDi_j keywords
    to represent pixel scaling, rotation, etc?
    */
    bool getCDMatrix() const { return getB("CDMatrix"); }

    /**
    Get @ref FitsChan_Clean "Clean": remove cards used whilst reading even if an error occurs?
    */
    bool getClean() const { return getB("Clean"); }

    /**
    Get @ref FitsChan_DefB1950 "DefB1950": use FK4 B1950 as default equatorial coordinates?
    */
    bool getDefB1950() const { return getB("DefB1950"); }

    /**
    Get @ref FitsChan_Encoding "Encoding": the encoding system to use when AST
    @ref Object "Objects" are stored as FITS header cards in a @ref FitsChan.
    */
    std::string getEncoding() const { return getC("Encoding"); }

    /**
    Get @ref FitsChan_FitsAxisOrder "FitsAxisOrder": the order for the WCS axes in any new
    FITS-WCS headers created using @ref Channel.write.
    */
    std::string getFitsAxisOrder() const { return getC("FitsAxisOrder"); }

    /**
    Get @ref FitsChan_FitsDigits "FitsDigits": digits of precision
    for floating-point FITS values.
    */
    int getFitsDigits() const { return getI("FitsDigits"); }

    /**
    Get @ref FitsChan_FitsTol "FitsTol": Tolerance used for writing a FrameSet using a foreign encoding
    */
    double getFitsTol() const { return getD("FitsTol"); }

    /**
    Get @ref FitsChan_Iwc "Iwc": add a Frame describing Intermediate World Coords?
    */
    bool getIwc() const { return getB("Iwc"); }

    /**
    Get @ref FitsChan_NCard "NCard": the number of cards
    */
    int getNCard() const { return getI("NCard"); }

    /**
    Get @ref FitsChan_Nkey "Nkey": the number of *unique* keywords
    */
    int getNKey() const { return getI("NKey"); }

    /**
    Get @ref FitsChan_SipOK "SipOK": use Spitzer Space Telescope keywords to define distortion?
    */
    bool getSipOK() const { return getB("SipOK"); }

    /**
    Get @ref FitsChan_SipReplace "SipReplace": ignore inverse SIP coefficients (replacing them with
    fit coefficients or an iterative inverse)?
    */
    bool getSipReplace() const { return getB("SipReplace"); }

    /**
    Get @ref FitsChan_TabOK "TabOK": should the FITS "-TAB" algorithm be recognised?
    */
    int getTabOK() const { return getI("TabOK"); }

    /**
    Get @ref FitsChan_PolyTan "PolyTan": use `PVi_m` keywords to define
    distorted TAN projection?
    */
    int getPolyTan() const { return getI("PolyTan"); }

    /**
    Get @ref FitsChan_Warnings "Warnings" attribute, which controls the issuing of warnings about
    selected conditions when an @ref Object or keyword is read from or written to a @ref FitsChan.
    */
    std::string getWarnings() const { return getC("Warnings"); }

    /**
    Delete all cards in a @ref FitsChan that relate to any of the recognised WCS encodings.

    On exit, the current card is the first remaining card in the @ref FitsChan.
    */
    void purgeWcs() {
        astPurgeWCS(getRawPtr());
        assertOK();
    }

    /**
    Replace all FITS header cards.

    The cards are supplied concatenated together into a single character string.
    Any existing cards in the @ref FitsChan are removed before the new cards
    are added. The @ref FitsChan is "re-wound" on exit.
    This means that a subsequent invocation of read can be made immediately
    without the need to re-wind the @ref FitsChan first.

    @param[in] cards  A string containing the FITS cards to be stored.
                    Each individual card should occupy 80 characters in this string,
                    and there should be no delimiters, new lines, etc, between adjacent cards.
                    The final card may be less than 80 characters long.
                    This is the format produced by the fits_hdr2str function in the
                    CFITSIO library.
    */
    void putCards(std::string const &cards) {
        astPutCards(getRawPtr(), cards.c_str());
        assertOK();
    }

    /**
    Store a FITS header card in a @ref FitsChan.

    The card is either inserted before the current card
    or over-writes the current card, depending on `overwrite`.

    @param[in] card  A character string containing the FITS cards to be stored.
                Each individual card should occupy 80 characters in this string,
                and there should be no delimiters, new lines, etc, between adjacent cards.
                The final card may be less than 80 characters long.
                This is the format produced by the fits_hdr2str function in the CFITSIO library.
    @param[in] overwrite   if `false`, the new card is inserted in before the current card.
                If `true` the new card replaces the current card.  In either case, the `Card`
                attribute is then incremented by one so that it subsequently identifies the card
                following the one stored.

    ### Notes

    - If the @ref FitsChan_Card "Card" attribute initially points at the "end-of-file"
        (i.e. exceeds the number of cards in the @ref FitsChan), then the new card
        is appended as the last card in the @ref FitsChan.
    - An error will result if the supplied string cannot be interpreted as a FITS header card.
    */
    void putFits(std::string const &card, bool overwrite) {
        astPutFits(getRawPtr(), card.c_str(), overwrite);
        assertOK();
    }

    /**
    Read cards from the source and store them in the @ref FitsChan.

    This normally happens once-only, when the @ref FitsChan is accessed for the first time.
    This function provides a means of forcing a re-read of the external source,
    and may be useful if (say) new cards have been deposited into the external source.
    Any new cards read from the source are appended to the end of the current contents of the @ref FitsChan.

    ### Notes

    -  This is a no-op if the @ref Stream has no source.
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
    inserted before the current card and the current card is left unchanged. In either case, if the current
    card on entry points to the "end-of-file", the new card is appended to the end of the list.
    */
    void setFitsCF(std::string const &name, std::complex<double> value, std::string const &comment = "",
                   bool overwrite = false) {
        // this use of reinterpret_cast is explicitly permitted, for C compatibility
        astSetFitsCF(getRawPtr(), name.c_str(), reinterpret_cast<double(&)[2]>(value), comment.c_str(),
                     overwrite);
        assertOK();
    }

    /**
    Create a new comment card, possibly overwriting the current card

    The card will have a name of "        " (eight spaces) and no equals sign.
    There is presently no way to generate a card with name HISTORY or COMMENT,
    but FITS treats those essentially the same as cards with blank names.

    @param[in] comment  Comment to associated with the keyword.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsCM(std::string const &comment, bool overwrite = false) {
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsCN(std::string const &name, std::string value, std::string const &comment = "",
                   bool overwrite = false) {
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsF(std::string const &name, double value, std::string const &comment = "",
                  bool overwrite = false) {
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsI(std::string const &name, int value, std::string const &comment = "",
                  bool overwrite = false) {
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsL(std::string const &name, bool value, std::string const &comment = "",
                  bool overwrite = false) {
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
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
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
    void setFitsS(std::string const &name, std::string value, std::string const &comment = "",
                  bool overwrite = false) {
        astSetFitsS(getRawPtr(), name.c_str(), value.c_str(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Create a new card with an undefined value, possibly overwriting the current card

    @param[in] name  Name of keyword for the new card.
            This may be a complete FITS header card, in which case the keyword to use
            is extracted from it.
    @param[in] comment  Comment to associated with the keyword. If blank, then any comment included
            in the string supplied for the `name` parameter is used instead. If `name`
            contains no comment, then any existing comment in the card being over-written
            is retained. Otherwise, no comment is stored with the card.
    @param[in] overwrite  if `true` the new card formed from the supplied keyword name,
            value and comment over-writes the current card, and the current card is incremented to refer
            to the next card (see the @ref FitsChan_Card "Card" attribute).  If `false`, the new card is
            inserted before the current card and the current card is left unchanged. In either
            case, if the current card on entry points to the "end-of-file", the new card
            is appended to the end of the list.

    ### Notes

    - To assign a new value for an existing keyword, first find the card describing the keyword
        using `findFits`, and then use the appropriate `setFits` function to over-write the old value.
    - If, on exit, there are no cards following the card written by this function,
        then the current card is left pointing at the "end-of-file" .
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    void setFitsU(std::string const &name, std::string const &comment = "", bool overwrite = false) {
        astSetFitsU(getRawPtr(), name.c_str(), comment.c_str(), overwrite);
        assertOK();
    }

    /**
    Set @ref FitsChan_CarLin "CarLin": ignore spherical rotations on CAR projections?
    */
    void setCarLin(bool cdMatrix) { setB("CarLin", cdMatrix); }

    /**
    Get @ref FitsChan_CDMatrix "CDMatrix": Use CDi_j keywords to represent pixel scaling,
    rotation, etc?
    */
    void setCDMatrix(bool cdMatrix) { setB("CDMatrix", cdMatrix); }

    /**
    Set @ref FitsChan_Clean "Clean": remove cards used whilst reading even if an error occurs?
    */
    void setClean(bool clean) { setB("Clean", clean); }

    /**
    Set @ref FitsChan_DefB1950 "DefB1950": use FK4 B1950 as default equatorial coordinates?
    */
    void setDefB1950(bool defB1950) { setB("DefB1950", defB1950); }

    /**
    Set @ref FitsChan_Encoding "Encoding": the encoding system to use when AST
    @ref Object "Objects" are stored as FITS header cards in a @ref FitsChan.
    */
    void setEncoding(std::string const &encoding) { setC("Encoding", encoding); }

    /**
    Set @ref FitsChan_FitsAxisOrder "FitsAxisOrder": the order for the WCS axes in any new
    FITS-WCS headers created using @ref Channel.write.
    */
    void setFitsAxisOrder(std::string const &order) { setC("FitsAxisOrder", order); }

    /**
    Set @ref FitsChan_FitsDigits "FitsDigits": digits of precision
    for floating-point FITS values
    */
    void setFitsDigits(int digits) { setI("FitsDigits", digits); }

    /**
    Set @ref FitsChan_FitsTol "FitsTol": Tolerance used for writing a FrameSet using a foreign encoding
    */
    void setFitsTol(double tol) { return setD("FitsTol", tol); }

    /**
    Set @ref FitsChan_Iwc "Iwc": add a Frame describing Intermediate World Coords?
    */
    void setIwc(bool iwcs) { setB("Iwc", iwcs); }

    /**
    Set @ref FitsChan_SipOK "SipOK": use Spitzer Space Telescope keywords to define distortion?
    */
    void setSipOK(bool sipOK) { setB("SipOK", sipOK); }

    /**
    Set @ref FitsChan_SipReplace "SipReplace": ignore inverse SIP coefficients (replacing them with
    fit coefficients or an iterative inverse)?
    */
    void setSipReplace(bool replace) { setB("SipReplace", replace); }

    /**
    Set @ref FitsChan_TabOK "TabOK": should the FITS "-TAB" algorithm be recognised?
    */
    void setTabOK(int tabOK) { setI("TabOK", tabOK); }

    /**
    Set @ref FitsChan_PolyTan "PolyTan": use `PVi_m` keywords
    to define distorted TAN projection?
    */
    void setPolyTan(int polytan) { setI("PolyTan", polytan); }

    /**
    Set @ref FitsChan_Warnings "Warnings" attribute, which controls the issuing of warnings about
    selected conditions when an @ref Object or keyword is read from or written to a @ref FitsChan.
    */
    void setWarnings(std::string const &warnings) { setC("Warnings", warnings); }

    /**
    Write all the cards in the channel to standard output
    */
    void showFits() const {
        astShowFits(getRawPtr());
        assertOK();
    }

    /**
    Determine if a card is present, and if so, whether it has a value.

    @param[in] name  Name of keyword, or empty for the current card

    ### Notes

    - This function does not change the current card.
    - If name is not empty then the card following the current card is checked first.
      If this is not the required card, then the rest of the @ref FitsChan is searched,
      starting with the first card added to the @ref FitsChan.
      Therefore cards should be accessed in the order they are stored in the @ref FitsChan (if possible)
      as this will minimise the time spent searching for cards.
    - An error will be reported if the keyword name does not conform to FITS requirements.
    */
    FitsKeyState testFits(std::string const &name = "") const;

    /**
    Write out all cards currently in the channel and clear the channel.
    */
    void writeFits() {
        astWriteFits(getRawPtr());
        assertOK();
    }

    /// Rewind the card index to the beginning
    void clearCard() { clear("Card"); }

    /**
    Set @ref FitsChan_Card "Card": the index of the current card, where 1 is the first card.
    */
    void setCard(int ind) { setI("Card", ind); }

    std::shared_ptr<KeyMap> getTables() const {
        auto *rawKeyMap = reinterpret_cast<AstObject *>(astGetTables(getRawPtr()));
        assertOK(rawKeyMap);
        if (!rawKeyMap) {
            throw std::runtime_error("getTables failed (returned a null keymap)");
        }
        return Object::fromAstObject<KeyMap>(rawKeyMap, true);
    }


    /**
    Construct a FitsChan from a raw AstFitsChan
    */
    explicit FitsChan(AstFitsChan *rawFitsChan) : Channel(reinterpret_cast<AstChannel *>(rawFitsChan)) {
        if (!astIsAFitsChan(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a FitsChan";
            throw std::invalid_argument(os.str());
        }
        assertOK();
    }
};

}  // namespace ast

#endif
