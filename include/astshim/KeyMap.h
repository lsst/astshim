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
#ifndef ASTSHIM_KEYMAP_H
#define ASTSHIM_KEYMAP_H

#include <complex>
#include <ostream>
#include <memory>

#include "astshim/base.h"
#include "astshim/Channel.h"
#include "astshim/Object.h"

namespace ast {

namespace {

void throwKeyNotFound(std::string const &key) {
    // make sure there isn't some other error, first
    assertOK();
    std::ostringstream os;
    os << "Key \"" << key << "\" not found or has an undefined value";
    throw std::runtime_error(os.str());
}

}  // namespace

/**
KeyMap is used to store a set of values with associated keys which identify the values.

The keys are strings. These may be case sensitive or insensitive as selected by the KeyCase attribute,
and trailing spaces are ignored. Each key is associated a vector of values of a particular type,
which is one of the following, where the single letter is the suffix for the associated
setter or getter, such as getD, putD or replaceD:
- D: double
- F: float
- I: int
- S: short int
- B: char
- C: string (internally: char *)
- A: astshim Object (internally: AstObject *)

Despite the name, KeyMap is not a Mapping.

The getters will attempt to cast the data into the requested form.
The setters come in three forms:
- put<X> that takes a scalar will and a key with one value
- put<X> that takes a vector will add a key with a vector of values
- append will append one value to a key
    (in AST this is one of two functions handled by `astMapPutElem<X>`
- replace will replace one value in a key
    (in AST this is one of two functions handled by `astMapPutElem<X>`)

### Attributes

KeyMap has the following attributes, in addition to those inherited from @ref Object

- `KeyCase`: Sets the case in which keys are stored
- `KeyError`: Report an error if the requested key does not exist?
- `SizeGuess`: The expected size of the KeyMap.
- `SortBy`: Determines how keys are sorted in a KeyMap.
- `MapLocked`: Prevent new entries being added to the KeyMap?
*/
class KeyMap : public Object {
    friend class Object;
    friend class Channel;

public:
    /**
    Construct an empty KeyMap

    @param[in] options  Comma-separated list of attribute assignments.
    */
    explicit KeyMap(std::string const &options = "")
            : Object(reinterpret_cast<AstObject *>(astKeyMap("%s", options.c_str()))) {
        assertOK();
    }

    virtual ~KeyMap(){};

    /// Copy constructor: make a deep copy
    KeyMap(KeyMap const &) = default;
    KeyMap(KeyMap &&) = default;
    KeyMap &operator=(KeyMap const &) = delete;
    KeyMap &operator=(KeyMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<KeyMap> copy() const {
        return std::static_pointer_cast<KeyMap>(copyPolymorphic());
        assertOK();
    }

    /**
    Does this map contain the specified key, and if so, does it have a defined value?

    See also hasKey, which does not check if the value is defined
    */
    bool defined(std::string const &key) const {
        bool ret = static_cast<bool>(
                astMapDefined(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str()));
        assertOK();
        return ret;
    }

    /// Get the key at the specified index
    std::string key(int ind) const {
        int const len = size();
        if ((ind < 0) || (ind >= len)) {
            std::ostringstream os;
            os << "ind = " << ind << " not in range [0, " << len - 1 << "]";
            throw std::invalid_argument(os.str());
        }
        const char *rawKey = astMapKey(reinterpret_cast<AstKeyMap const *>(getRawPtr()), ind);
        assertOK();
        return std::string(rawKey);
    }

    /**
    Does this map contain the specified key?

    See also defined, which also checks that the value is defined
    */
    bool hasKey(std::string const &key) const {
        bool ret = static_cast<bool>(
                astMapHasKey(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str()));
        assertOK();
        return ret;
    }

    /// Get the size of the vector for the specified key; return 0 if key not found or value is undefined
    int length(std::string const &key) const {
        int len = astMapLength(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str());
        assertOK();
        return len;
    }

    /// Get the number of keys
    int size() const {
        int const size = astMapSize(reinterpret_cast<AstKeyMap const *>(getRawPtr()));
        assertOK();
        return size;
    }

    /// Get one double value for a given key
    double getD(std::string const &key, int ind) const {
        double retVal;
        if (!astMapGetElemD(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &retVal)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return retVal;
    }

    /// Get all double values for a given key
    std::vector<double> getD(std::string const &key) const {
        int const size = length(key);
        std::vector<double> retVec(size);
        if (size > 0) {
            int nret;  // should equal size after the call
            astMapGet1D(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), size, &nret,
                        retVec.data());
        }
        assertOK();
        return retVec;
    }

    /// Get one float value for a given key
    float getF(std::string const &key, int ind) const {
        float retVal;
        if (!astMapGetElemF(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &retVal)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return retVal;
    }

    /// Get all float values for a given key
    std::vector<float> getF(std::string const &key) const {
        int const size = length(key);
        std::vector<float> retVec(size);
        if (size > 0) {
            int nret;  // should equal size after the call
            astMapGet1F(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), size, &nret,
                        retVec.data());
        }
        assertOK();
        return retVec;
    }

    /// Get one int value for a given key
    int getI(std::string const &key, int ind) const {
        int retVal;
        if (!astMapGetElemI(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &retVal)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return retVal;
    }

    /// Get all int values for a given key
    std::vector<int> getI(std::string const &key) const {
        int const size = length(key);
        std::vector<int> retVec(size);
        if (size > 0) {
            int nret;  // should equal size after the call
            astMapGet1I(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), size, &nret,
                        retVec.data());
        }
        assertOK();
        return retVec;
    }

    /// Get one short int value for a given key
    short int getS(std::string const &key, int ind) const {
        short int retVal;
        if (!astMapGetElemS(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &retVal)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return retVal;
    }

    /// Get all short int values for a given key
    std::vector<short int> getS(std::string const &key) const {
        int const size = length(key);
        std::vector<short int> retVec(size);
        if (size > 0) {
            int nret;  // should equal size after the call
            astMapGet1S(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), size, &nret,
                        retVec.data());
        }
        assertOK();
        return retVec;
    }

    /// Get one char value for a given key
    char unsigned getB(std::string const &key, int ind) const {
        char unsigned retVal;
        if (!astMapGetElemB(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &retVal)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return retVal;
    }

    /// Get all char values for a given key
    std::vector<char unsigned> getB(std::string const &key) const {
        int const size = length(key);
        std::vector<char unsigned> retVec(size);
        if (size > 0) {
            int nret;  // should equal size after the call
            astMapGet1B(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), size, &nret,
                        retVec.data());
        }
        assertOK();
        return retVec;
    }

    /// Get one std::string value for a given key
    std::string getC(std::string const &key, int ind) const {
        int const maxChar = 1 + astMapLenC(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str());
        std::unique_ptr<char[]> cstr(new char[maxChar]);
        if (!astMapGetElemC(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), maxChar, ind,
                            cstr.get())) {
            throwKeyNotFound(key);
        }
        assertOK();
        return std::string(cstr.get());
    }

    /// Get all std::string values for a given key
    std::vector<std::string> getC(std::string const &key) const {
        int const size = length(key);
        std::vector<std::string> retVec;
        if (size > 0) {
            // # of chars for each entry; the +1 is needed to provide space for the terminating null
            int const eltLen = 1 + astMapLenC(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str());
            std::unique_ptr<char[]> cstr(new char[size * eltLen]);
            int nret;  // should equal size after the call
            astMapGet1C(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), eltLen, size, &nret,
                        cstr.get());
            for (int i = 0; i < size; ++i) {
                retVec.push_back(std::string(cstr.get() + i * eltLen));
            }
        }
        assertOK();
        return retVec;
    }

    /// Get one Object for a given key; the object is deep copied
    std::shared_ptr<Object> getA(std::string const &key, int ind) const {
        std::shared_ptr<Object> retVal;
        AstObject *rawObj;
        if (!astMapGetElemA(reinterpret_cast<AstKeyMap const *>(getRawPtr()), key.c_str(), ind, &rawObj)) {
            throwKeyNotFound(key);
        }
        assertOK();
        return Object::fromAstObject<Object>(rawObj, true);
    }

    /// Get all Objects for a given key; each object is deep copied
    std::vector<std::shared_ptr<Object>> getA(std::string const &key) const {
        int const size = length(key);
        std::vector<std::shared_ptr<Object>> retVec;
        for (int i = 0; i < size; ++i) {
            retVec.push_back(getA(key, i));
        }
        return retVec;
    }

    /// Add a double value
    void putD(std::string const &key, double value, std::string const &comment = "") {
        astMapPut0D(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value, comment.c_str());
        assertOK();
    }

    /// Add a vector of double
    void putD(std::string const &key, std::vector<double> const &vec, std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        astMapPut1D(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), vec.size(), vec.data(),
                    comment.c_str());
        assertOK();
    }

    /// Add a float
    void putF(std::string const &key, float value, std::string const &comment = "") {
        astMapPut0F(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value, comment.c_str());
        assertOK();
    }

    /// Add a vector of floats
    void putF(std::string const &key, std::vector<float> const &vec, std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        astMapPut1F(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), vec.size(), vec.data(),
                    comment.c_str());
        assertOK();
    }

    /// Add an int
    void putI(std::string const &key, int value, std::string const &comment = "") {
        astMapPut0I(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value, comment.c_str());
        assertOK();
    }

    /// Add a vector of ints
    void putI(std::string const &key, std::vector<int> const &vec, std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        astMapPut1I(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), vec.size(), vec.data(),
                    comment.c_str());
        assertOK();
    }

    /// Add a short int
    void putS(std::string const &key, short int value, std::string const &comment = "") {
        astMapPut0S(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value, comment.c_str());
        assertOK();
    }

    /// Add a vector of short int
    void putS(std::string const &key, std::vector<short int> const &vec, std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        astMapPut1S(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), vec.size(), vec.data(),
                    comment.c_str());
        assertOK();
    }

    /// Add a char
    void putB(std::string const &key, char unsigned value, std::string const &comment = "") {
        astMapPut0B(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value, comment.c_str());
        assertOK();
    }

    /// Add a vector of chars
    void putB(std::string const &key, std::vector<char unsigned> const &vec,
              std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        astMapPut1B(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), vec.size(), vec.data(),
                    comment.c_str());
        assertOK();
    }

    /// Add a string
    void putC(std::string const &key, std::string const &value, std::string const &comment = "") {
        astMapPut0C(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), value.c_str(), comment.c_str());
        assertOK();
    }

    /// Add a vector of strings
    void putC(std::string const &key, std::vector<std::string> const &vec, std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        // to simplify memory management, create the key with the first element and append the rest
        for (int i = 0, size = vec.size(); i < size; ++i) {
            if (i == 0) {
                putC(key, vec[0]);
            } else {
                append(key, vec[i]);
            }
        }
    }

    /// Add an Object, which is deep copied
    void putA(std::string const &key, Object const &obj, std::string const &comment = "") {
        AstObject *rawCopy = reinterpret_cast<AstObject *>(astCopy(obj.getRawPtr()));
        astMapPut0A(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), rawCopy, comment.c_str());
        assertOK(rawCopy);
    }

    /// Add a vector of shared pointer to Object; the objects are deep copied
    void putA(std::string const &key, std::vector<std::shared_ptr<Object const>> const &vec,
              std::string const &comment = "") {
        _assertVectorNotEmpty(key, vec.size());
        // to simplify memory management, create the key with the first element and append the rest
        for (int i = 0, size = vec.size(); i < size; ++i) {
            if (i == 0) {
                // initialize the key with the first element
                putA(key, *vec[0]);
            } else {
                append(key, *vec[i]);
            }
        }
    }

    /**
    Add a new entry, but no value is stored with the entry.

    The entry has a special data type represented by symbolic constant AST__UNDEFTYPE.
    Such entries can act as placeholders for values that can be added to the KeyMap later.
    */
    void putU(std::string const &key, std::string const &comment = "") {
        astMapPutU(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), comment.c_str());
        assertOK();
    }

    /// Append an element to a vector of doubles in a KeyMap
    void append(std::string const &key, double value) {
        int const i = _getAppendIndex(key);
        astMapPutElemD(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Append an element to a vector of floats in a KeyMap
    void append(std::string const &key, float value) {
        int const i = _getAppendIndex(key);
        astMapPutElemF(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Append an element to a vector of ints in a KeyMap
    void append(std::string const &key, int value) {
        int const i = _getAppendIndex(key);
        astMapPutElemI(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Append an element to a vector of short int in a KeyMap
    void append(std::string const &key, short int value) {
        int const i = _getAppendIndex(key);
        astMapPutElemS(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Append an element to a vector of char in a KeyMap
    void append(std::string const &key, char unsigned value) {
        int const i = _getAppendIndex(key);
        astMapPutElemB(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Append an element to a vector of strings in a KeyMap
    void append(std::string const &key, std::string const &value) {
        int const i = _getAppendIndex(key);
        astMapPutElemC(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value.c_str());
        assertOK();
    }

    /// Append an element to a vector of Objects in a KeyMap
    void append(std::string const &key, Object const &value) {
        int const i = _getAppendIndex(key);
        AstObject *rawCopy = reinterpret_cast<AstObject *>(astCopy(value.getRawPtr()));
        astMapPutElemA(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, rawCopy);
        assertOK(rawCopy);
    }

    /// Replace an element of a vector of doubles in a KeyMap
    void replace(std::string const &key, int i, double value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemD(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Replace an element of a vector of floats in a KeyMap
    void replace(std::string const &key, int i, float value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemF(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Replace an element of a vector of ints in a KeyMap
    void replace(std::string const &key, int i, int value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemI(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Replace an element of a vector of short int in a KeyMap
    void replace(std::string const &key, int i, short int value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemS(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Replace an element of a vector of char in a KeyMap
    void replace(std::string const &key, int i, char unsigned value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemB(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value);
        assertOK();
    }

    /// Replace an element of a vector of strings in a KeyMap
    void replace(std::string const &key, int i, std::string const &value) {
        _assertReplaceIndexInRange(key, i);
        astMapPutElemC(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, value.c_str());
        assertOK();
    }

    /// Replace an element of a vector of Objects in a KeyMap
    void replace(std::string const &key, int i, Object const &value) {
        _assertReplaceIndexInRange(key, i);
        AstObject *rawCopy = reinterpret_cast<AstObject *>(astCopy(value.getRawPtr()));
        astMapPutElemA(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str(), i, rawCopy);
        assertOK(rawCopy);
    }

    /**
    Remove the specified entry.

    Silently do nothing if this KeyMap does not contain the specified key.
    */
    void remove(std::string const &key) {
        astMapRemove(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str());
        assertOK();
    }

    /**
    Rename the specified entry.

    Silently do nothing if this KeyMap does not contain the old key.
    */
    void rename(std::string const &oldKey, std::string const &newKey) {
        astMapRename(reinterpret_cast<AstKeyMap *>(getRawPtr()), oldKey.c_str(), newKey.c_str());
        assertOK();
    }

    /**
    Get the type suffix for a given key
    */
    DataType type(std::string const &key) {
        int retVal = astMapType(reinterpret_cast<AstKeyMap *>(getRawPtr()), key.c_str());
        assertOK();
        return static_cast<DataType>(retVal);
    }

protected:
    // Protected implementation of deep-copy.
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return std::static_pointer_cast<KeyMap>(copyImpl<KeyMap, AstKeyMap>());
    }

    /**
    Construct a KeyMap from a raw AstKeyMap
    */
    explicit KeyMap(AstKeyMap *rawKeyMap) : Object(reinterpret_cast<AstObject *>(rawKeyMap)) {
        if (!astIsAKeyMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a KeyMap";
            throw std::invalid_argument(os.str());
        }
        assertOK();
    }

private:
    void _assertVectorNotEmpty(std::string const &key, int size) const {
        if (size == 0) {
            std::ostringstream os;
            os << "vector supplied for key \"" << key << "\" has zero elements";
            throw std::invalid_argument(os.str());
        }
    }

    // replace silently fails if index out of range, so check it here
    void _assertReplaceIndexInRange(std::string const &key, int i) const {
        int const len = length(key);
        if ((i < 0) || (i >= len)) {
            std::ostringstream os;
            os << "i = " << i << " not in range [0, " << len - 1 << "] for key \"" << key << "\"";
            throw std::invalid_argument(os.str());
        }
    }

    // retrieve the index required to append a value to a key, and make sure the key exists
    int _getAppendIndex(std::string const &key) const {
        int const i = length(key);
        if (i == 0) {
            std::ostringstream os;
            os << "key \"" << key << "\" not found";
            throw std::invalid_argument(os.str());
        }
        return i;
    }
};

}  // namespace ast

#endif
