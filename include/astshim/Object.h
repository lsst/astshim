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
#ifndef ASTSHIM_OBJECT_H
#define ASTSHIM_OBJECT_H

#include <ostream>
#include <memory>

#include "astshim/base.h"
#include "astshim/detail/utils.h"
#include "astshim/MapSplit.h"

namespace ast {

class FrameDict;

/**
Abstract base class for all AST objects

### Attributes

Object provides the following attributes:

- @ref Object_Class "Class": object class name (use @ref getClassName)
- @ref Object_ID "ID": object identification string that is not copied.
- @ref Object_Ident "Ident": object identification string that is copied.
- @ref Object_NObject "NObject": number of Objects in class
- @ref Object_ObjSize "ObjSize": the in-memory size of the Object in bytes
- @ref Object_RefCount "RefCount": count of active Object pointers
- @ref Object_UseDefs "UseDefs": allow use of default values for Object attributes?
*/
class Object {
    friend class MapSplit;
    friend class FrameDict;

private:
    using Deleter = void (*)(AstObject *);

public:
    using ObjectPtr = std::unique_ptr<AstObject, Deleter>;  ///< unique pointer holding an AST raw pointer

    virtual ~Object() {}

    /// Copy constructor: make a deep copy
    Object(Object const &object) : _objPtr(object.getRawPtrCopy(), &detail::annulAstObject) {}
    Object(Object &&) = default;
    Object &operator=(Object const &) = delete;
    Object &operator=(Object &&) = default;

    /**
    Return True if this and `rhs` are the equal

    For two objects be equal, they both must have the same attributes and all contained objects
    must be equal.
    */
    bool operator==(Object const &rhs) const;

    /**
    Return True if this and `rhs` are not equal

    See operator== for details
    */
    bool operator!=(Object const &rhs) const {
        return !(*this == rhs); };

    /**
    Construct an @ref Object from a string, using astFromString
    */
    static std::shared_ptr<Object> fromString(std::string const &str) {
        auto *rawPtr = reinterpret_cast<AstObject *>(astFromString(str.c_str()));
        assertOK(rawPtr);
        return Object::_basicFromAstObject(rawPtr);
    }

    /**
    Given a bare AST object pointer return a shared pointer to an ast::Object of the correct type

    The returned object takes ownership of the pointer. This is almost always what you want,
    for instance astDecompose returns shallow copies of the internal pointers.

    @tparam Class  The class of the returned shared pointer. (The actual class will be
                   the correct class of rawPtr.)
    @param[in] rawObj  A bare AST object pointer
    @param[in] copy  If True then make a deep copy of the pointer (and free the original)
    */
    template <typename Class>
    static std::shared_ptr<Class> fromAstObject(AstObject *rawObj, bool copy);

    /// Return a deep copy of this object.
    std::shared_ptr<Object> copy() const { return std::static_pointer_cast<Object>(copyPolymorphic()); }

    /**
    Clear the values of a specified set of attributes for an Object.

    Clearing an attribute cancels any value that has previously been explicitly set for it,
    so that the standard default attribute value will subsequently be used instead.
    This also causes the astTest function to return the value zero for the attribute,
    indicating that no value has been set.
    */
    void clear(std::string const &attrib) {
        astClear(getRawPtr(), attrib.c_str());
        assertOK();
    }

    /**
    Does this object have an attribute with the specified name?
    */
    bool hasAttribute(std::string const &attrib) const {
        bool ret = astHasAttribute(getRawPtr(), attrib.c_str());
        assertOK();
        return ret;
    }

    /**
    Get @ref Object_Class "Class": the name of the class (e.g. ZoomMap)

    Note: if AST returns "CmpMap" then the name will be changed
    to "SeriesMap" or "ParallelMap", as appropriate.
    */
    std::string getClassName() const { return detail::getClassName(getRawPtr()); }

    /// Get @ref Object_ID "ID": object identification string that is not copied.
    std::string getID() const { return getC("ID"); }

    /// Get @ref Object_Ident "Ident": object identification string that is copied.
    std::string getIdent() const { return getC("Ident"); }

    /**
    Get @ref Object_NObject "NObject": number of AST objects in existence of the same type
    as the underlying AST class.

    @warning Intended only for debugging astshim.
    */
    int getNObject() const { return getI("NObject"); }

    /// Get @ref Object_ObjSize "ObjSize": the in-memory size of the AST object in bytes.
    int getObjSize() const { return getI("ObjSize"); }

    /**
    Get @ref Object_RefCount "RefCount": number of active pointers to the underlying AST object

    @warning Intended only for debugging astshim.
    */
    int getRefCount() const { return getI("RefCount"); }

    /// Get @ref Object_UseDefs "UseDefs": allow use of default values for Object attributes?
    bool getUseDefs() const { return getB("UseDefs"); }

    /**
    Lock this object for exclusive use by the calling thread.

    The thread-safe public interface to AST is designed so that an
    error is reported if any thread attempts to use an @ref Object that it
    has not previously locked for its own exclusive use using this
    function. When an @ref Object is created, it is initially locked by the
    thread that creates it, so newly created objects do not need to be
    explicitly locked. However, if an @ref Object pointer is passed to
    another thread, the original thread must first unlock it (using
    astUnlock) and the new thread must then lock it (using astLock)
    before the new thread can use the @ref Object.

    @param[in] wait  If the @ref Object is curently locked by another thread then this
       function will either report an error or block. If a non-zero value
       is supplied for "wait", the calling thread waits until the object
       is available for it to use. Otherwise, an error is reported and
       the function returns immediately without locking the @ref Object.

    ### Notes

    - The Locked object will belong to the current AST context.
    - This function returns without action if the @ref Object is already
    locked by the calling thread.
    - If simultaneous use of the same object is required by two or more
    threads, @ref Object.copy should be used to to produce a deep copy of
    the @ref Object for each thread. Each copy should then be unlocked by
    the parent thread (i.e. the thread that created the copy), and then
    locked by the child thread (i.e. the thread that wants to use the
    copy).
    - This function returns without action if the AST library has
    been built without POSIX thread support (i.e. the "-with-pthreads"
    option was not specified when running the "configure" script).
    */
    void lock(bool wait) {
        astLock(getRawPtr(), static_cast<int>(wait));
        assertOK();
    }

    /**
    Does this contain the same AST object as another?

    This is a test of identity, not of equality.
    */
    bool same(Object const &other) const { return astSame(getRawPtr(), other.getRawPtr()); }

    /// Set @ref Object_ID "ID": object identification string that is not copied.
    void setID(std::string const &id) { setC("ID", id); }

    /// Set @ref Object_Ident "Ident": object identification string that is copied.
    void setIdent(std::string const &ident) { setC("Ident", ident); }

    /// Set @ref Object_UseDefs "UseDefs": allow use of default values for Object attributes?
    void setUseDefs(bool usedefs) { setB("UseDefs", usedefs); }

    /**
    Print a textual description the object to an ostream.

    @param[in, out] os  The stream to which to write the string representation.
    @param[in] showComments  Show comments?
    */
    void show(std::ostream &os, bool showComments = true) const;

    /**
    Return a textual description the object as a string.

    @param[in] showComments  Show comments?
    */
    std::string show(bool showComments = true) const;

    /**
    Has this attribute been explicitly set (and not subsequently cleared)?

    @warning Unlike the underlying astTest function, throws an exception if an error results

    ### Notes

    - Attribute names are not case sensitive and may be surrounded by white space.
    - As you might expect, the returned value for a read-only attribute is always `false`.

    @throws std::runtime_error if an error results.
    */
    bool test(std::string const &attrib) const {
        bool res = astTest(getRawPtr(), attrib.c_str());
        assertOK();
        return res;
    }

    /**
    Unlock this object previously locked using @ref lock, so that other
    threads can use this object. See @ref lock for further details.

    @param[in] report  If true, an error will be reported if the supplied @ref Object,
       or any @ref Object contained within the supplied @ref Object, is not
       currently locked by the running thread. If false, such @ref Object "Objects"
       will be left unchanged, and no error will be reported.

    ### Notes

    - This function attempts to execute even if AST's global error
    status is set, but no further error report will be made if it
    subsequently fails under these circumstances.
    - All unlocked @ref Object "Objects" are excluded from AST context handling until
    they are re-locked using astLock.
    - This function returns without action if the @ref Object is not currently
    locked by any thread. If it is locked by the running thread, it is
    unlocked. If it is locked by another thread, an error will be reported
    if "error" is non-zero.
    - This function returns without action if the AST library has
    been built without POSIX thread support (i.e. the "-with-pthreads"
    option was not specified when running the "configure" script).
    */
    void unlock(bool report = false) {
        astUnlock(getRawPtr(), static_cast<int>(report));
        assertOK();
    }

    /**
    Get the raw AST pointer.

    Intended for internal use only, but cannot be made protected
    without endless "friend class" declarations.
    @{
    */
    AstObject const *getRawPtr() const { return &*_objPtr; };

    AstObject *getRawPtr() { return &*_objPtr; };
    ///@}

protected:
    /**
    Construct an @ref Object from a pointer to a raw AstObject
    */
    explicit Object(AstObject *object);

    /**
    Functor to make an astshim instance from a raw AST pointer of the corresponding type.

    @tparam ShimT  Output astshim class
    @tparam AstT  Output AST class
    */
    template <typename ShimT, typename AstT>
    static std::shared_ptr<ShimT> makeShim(AstObject *p) {
        return std::shared_ptr<ShimT>(new ShimT(reinterpret_cast<AstT *>(p)));
    }

    /**
    Implementation of deep copy

    Should be called to implement copyPolymorphic by all derived classes.
    */
    template <typename T, typename AstT>
    std::shared_ptr<T> copyImpl() const {
        auto *rawptr = reinterpret_cast<AstT *>(astCopy(getRawPtr()));
        auto retptr = std::shared_ptr<T>(new T(rawptr));
        assertOK();
        return retptr;
    }

    /**
    Return a deep copy of this object. This is called by @ref copy.

    Each subclass must override this method. The standard implementation is:
    ```
        return copyImpl<astshim_class, ast_class>();
    ```
    for example @ref Frame implements this as:
    ```
        return copyImpl<Frame, AstFrame>();
    ```
    */
    virtual std::shared_ptr<Object> copyPolymorphic() const = 0;

    /**
    Get the value of an attribute as a bool

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    bool getB(std::string const &attrib) const {
        bool val = astGetI(getRawPtr(), attrib.c_str());
        assertOK();
        return val;
    }

    /**
    Get the value of an attribute as a string

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    std::string const getC(std::string const &attrib) const {
        char const *rawval = astGetC(getRawPtr(), attrib.c_str());
        assertOK();
        return std::string(rawval);
    }

    /**
    Get the value of an attribute as a double

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    double getD(std::string const &attrib) const {
        double val = astGetD(getRawPtr(), attrib.c_str());
        assertOK();
        return val;
    }

    /**
    Get the value of an attribute as a float

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    float getF(std::string const &attrib) const {
        float val = astGetF(getRawPtr(), attrib.c_str());
        assertOK();
        return val;
    }

    /**
    Get the value of an attribute as an int

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    int getI(std::string const &attrib) const {
        int val = astGetI(getRawPtr(), attrib.c_str());
        assertOK();
        return val;
    }

    /**
    Get the value of an attribute as a long int

    If possible, the attribute value is converted to the type you request.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    long int getL(std::string const &attrib) const {
        long int val = astGetL(getRawPtr(), attrib.c_str());
        assertOK();
        return val;
    }

    /**
    Assign a set of attribute values, over-riding any previous values.

    The attributes and their new values are specified via a character string,
    which should contain a comma-separated list of the form:
        "attribute_1 = value_1, attribute_2 = value_2, ... "
    where "attribute_n" specifies an attribute name, and the value to the right of each " =" sign
    should be a suitable textual representation of the value to be assigned.
    This value will be interpreted according to the attribute's data type.

    ### Notes

    - Attribute names are not case sensitive and may be surrounded by white space
    - Attribute names are not case sensitive and may be surrounded by white space.
    - White space may also surround attribute values, where it will generally be ignored (except for
      string-valued attributes where it is significant and forms part of the value to be assigned).
    - To include a literal comma or percent sign in the value assigned to an attribute,
      the whole attribute value should be enclosed in quotation markes.

    @throws std::runtime_error if the attribute is read-only
    */
    void set(std::string const &setting) {
        astSet(getRawPtr(), "%s", setting.c_str());
        assertOK();
    }

    /**
    Set the value of an attribute as a bool

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setB(std::string const &attrib, bool value) {
        astSetI(getRawPtr(), attrib.c_str(), value);
        assertOK();
    }

    /**
    Set the value of an attribute as a string

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setC(std::string const &attrib, std::string const &value) {
        astSetC(getRawPtr(), attrib.c_str(), value.c_str());
        assertOK();
    }

    /**
    Set the value of an attribute as a double

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setD(std::string const &attrib, double value) {
        astSetD(getRawPtr(), attrib.c_str(), value);
        assertOK();
    }

    /**
    Set the value of an attribute as a float

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setF(std::string const &attrib, float value) {
        astSetF(getRawPtr(), attrib.c_str(), value);
        assertOK();
    }

    /**
    Set the value of an attribute as an int

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setI(std::string const &attrib, int value) {
        astSetI(getRawPtr(), attrib.c_str(), value);
        assertOK();
    }

    /**
    Set the value of an attribute as a long int

    If possible, the type you provide is converted to the actual type of the attribute.

    @throws std::runtime_error if the attribute does not exist or the value cannot be converted
    */
    void setL(std::string const &attrib, long int value) {
        astSetL(getRawPtr(), attrib.c_str(), value);
        assertOK();
    }
private:
    /*
    Given a bare AST object pointer return a shared pointer to an ast::Object of the correct type

    The returned object takes ownership of the pointer. This is almost always what you want,
    for instance astDecompose returns shallow copies of the internal pointers.

    @param[in] rawObj  A bare AST object pointer
    */
    static std::shared_ptr<Object> _basicFromAstObject(AstObject *rawObj);

    /*
    Get a deep copy of the raw AST pointer.
    */
    AstObject * getRawPtrCopy() const {
        AstObject * rawPtrCopy = reinterpret_cast<AstObject *>(astCopy(getRawPtr()));
        assertOK(rawPtrCopy);
        return rawPtrCopy;
    }

    /*
    Swap the raw object pointers between this and another object
    */
    void swapRawPointers(Object &other) noexcept {
        swap(_objPtr, other._objPtr);
    }

    ObjectPtr _objPtr;
};

}  // namespace ast

#endif
