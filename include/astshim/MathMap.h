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
#ifndef ASTSHIM_MATHMAP_H
#define ASTSHIM_MATHMAP_H

#include <memory>

#include "astshim/base.h"
#include "astshim/Mapping.h"

namespace ast {

/**
A MathMap is a @ref Mapping which allows you to specify a set of forward and/or inverse transformation
functions using arithmetic operations and mathematical functions similar to those available in C.
The MathMap interprets these functions at run-time, whenever its forward or inverse transformation
is required. Because the functions are not compiled in the normal sense (unlike an IntraMap),
they may be used to describe coordinate transformations in a transportable manner.
A MathMap therefore provides a flexible way of defining new types of @ref Mapping whose descriptions
may be stored as part of a dataset and interpreted by other programs.

### Notes

- The sequence of numbers produced by the random number functions available within a MathMap
     is normally unpredictable and different for each MathMap. However, this behaviour may be controlled
     by means of the MathMap_Seed "Seed" attribute.
- Normally, compound mappings which involve MathMaps will not be subject to simplification
     because AST cannot know how different MathMaps will interact. However, in the special case
     where a MathMap occurs in series with its own inverse, then simplification may be possible.
     Whether simplification does, in fact, occur under these circumstances is controlled by
     the MathMap_SimpFI "SimpFI" and MathMap_SimpFI "SimpFI" attributes.

### Attributes

In addition to those attributes provided by @ref Mapping and @ref Object,
@ref MathMap has the following attributes:

- @ref MathMap_Seed "Seed": Random number seed
- @ref MathMap_SimpFI "SimpFI": Forward-inverse MathMap pairs simplify?
- @ref MathMap_SimpIF "SimpIF": Inverse-forward MathMap pairs simplify?
*/
class MathMap : public Mapping {
    friend class Object;

public:
    /**
    Construct a Mathmap

    @param[in] nin  Number of input variables for the MathMap.
    @param[in] nout  Number of output variables for the MathMap.
    @param[in] fwd  An array contain the expressions defining the forward transformation.
                    The syntax of these expressions is described below.
    @param[in] rev  An array contain the expressions defining the reverse transformation.
                    The syntax of these expressions is described below.
    @param[in] options  Comma-separated list of attribute assignments.

    ## Defining Transformation Functions:

    A `MathMap`'s transformation functions are supplied as a set of expressions in an array
    of character strings.  Normally you would supply the same number of expressions for
    the forward transformation, via the `fwd` parameter, as there are output variables
    (given by the `MathMap`'s NOut attribute).  For instance, if NOut is 2 you might use:
    - "r = sqrt(x*x + y*y)"
    - "theta = atan2(y, x)"
    which defines a transformation from Cartesian to polar coordinates.  Here, the variables
    that appear on the left of each expression (`r` and `theta` ) provide names for the
    output variables and those that appear on the right (`x` and `y`) are references
    to input variables.

    To complement this, you must also supply expressions for the inverse transformation
    via the "inv" parameter.  In this case, the number of expressions given would normally
    match the number of MathMap input coordinates (given by the NIn attribute). If NIn
    is 2, you might use:
    - "x = r * cos(theta)"
    - "y = r * sin(theta)"
    which expresses the transformation from polar to Cartesian coordinates.  Note that
    here the input variables (`x` and `y`) are named on the left of each expression,
    and the output variables (`r` and `theta` are referenced on the right.

    Normally, you cannot refer to a variable on the right of an expression unless it is
    named on the left of an expression in the complementary set of functions.  Therefore
    both sets of functions (forward and inverse) must be formulated using the same consistent
    set of variable names.  This means that if you wish to leave one of the transformations
    undefined, you must supply dummy expressions which simply name each of the output (or
    input) variables.  For example, you might use:
    - `x`
    - " y"
    for the inverse transformation above, which serves to name the input variables but
    without defining an inverse transformation.

    ### Calculating Intermediate Values:

    It is sometimes useful to calculate intermediate values and then to use these in the
    final expressions for the output (or input) variables.  This may be done by supplying
    additional expressions for the forward (or inverse) transformation functions.  For
    instance, the following array of five expressions describes 2-dimensional pin-cushion
    distortion:
    - "r=sqrt(xin * xin + yin * yin)"
    - "rout = r * (1 + 0.1 * r * r)"
    - "theta = atan2(yin, xin)"
    - "xout=rout * cos(theta)"
    - "yout=rout * sin(theta)"

    Here, we first calculate three intermediate results ("r" , " rout" and "theta" )
    and then use these to calculate the final results ("xout" and "yout" ).  The MathMap
    knows that only the final two results constitute values for the output variables because
    its NOut attribute is set to 2.  You may define as many intermediate variables in this
    way as you choose.  Having defined a variable, you may then refer to it on the right
    of any subsequent expressions.

    Note that when defining the inverse transformation you may only refer to the output
    variables "xout" and " yout" .  The intermediate variables "r", "rout" and "theta"
    (above) are private to the forward transformation and may not be referenced by the
    inverse transformation.  The inverse transformation may, however, define its own private
    intermediate variables.

    ### Expression Syntax:

    The expressions given for the forward and inverse transformations closely follow the
    syntax of the C programming language (with some extensions for compatibility with Fortran).
    They may contain references to variables and literal constants, together with arithmetic,
    boolean, relational and bitwise operators, and function invocations.  A set of symbolic
    constants is also available.  Each of these is described in detail below.  Parentheses
    may be used to over-ride the normal order of evaluation.  There is no built-in limit
    to the length of expressions and they are insensitive to case or the presence of additional
    white space.

    ### Variables:

    Variable names must begin with an alphabetic character and may contain only alphabetic
    characters, digits, and the underscore character " _" .  There is no built-in limit
    to the length of variable names.

    ### Literal Constants:

    Literal constants, such as `0`, `1`, `0.007` or `2.505e-16` may appear in expressions,
    with the decimal point and exponent being optional (a `D` may also be used as an exponent
    character for compatibility with Fortran).  A unary minus `-` may be used as a prefix.

    Arithmetic Precision:

    All arithmetic is floating point, performed in double precision.

    ### Propagation of Missing Data:

    Unless indicated otherwise, if any argument of a function or operator has the value AST__BAD
    (indicating missing data), then the result of that function or operation is also AST__BAD,
    so that such values are propagated automatically through all operations performed by MathMap
    transformations.
    The special value AST__BAD can be represented in expressions by the symbolic constant `<bad>`.
    A `<bad>` result (i.e. equal to AST__BAD) is also produced in response to any numerical error
    (such as division by zero or numerical overflow), or if an invalid argument value is provided to
    a function or operator.

    ### Arithmetic Operators:

    The following arithmetic operators are available:
    - `x1 + x2`: `x1` plus `x2`.
    - `x1 - x2`: `x1` minus `x2`.
    - `x1 * x2`: `x1` times `x2`.
    - `x1 / x2`: `x1` divided by `x2`.
    - `x1 ** x2`: `x1` raised to the power of `x2`.
    - `+x`: Unary plus, has no effect on its argument.
    - `-x`: Unary minus, negates its argument.

    ### Boolean Operators:

    Boolean values are represented using zero to indicate false and non-zero to indicate true.
    In addition, the value AST__BAD is taken to mean `unknown` . The values returned by boolean operators
    may therefore be 0, 1 or AST__BAD. Where appropriate, "tri-state" logic is implemented.
    For example, `a||b` may evaluate to 1 if `a` is non-zero, even if `b` has the value AST__BAD.
    This is because the result of the operation would not be affected by the value of `b`,
    so long as `a` is non-zero.

    The following boolean operators are available:
    - `x1 && x2`: Boolean AND between `x1` and `x2`, returning 1 if both `x1` and `x2` are non-zero,
         and 0 otherwise. This operator implements tri-state logic. (The synonym " .and." is also provided
         for compatibility with Fortran.)
    - `x1 || x2`: Boolean OR between `x1` and `x2`, returning 1 if either `x1` or `x2` are non-zero,
         and 0 otherwise. This operator implements tri-state logic. (The synonym " .or." is also provided
         for compatibility with Fortran.)
    - `x1 ^^ x2`: Boolean exclusive OR (XOR) between `x1` and `x2`, returning 1 if exactly one of `x1` and
         `x2` is non-zero, and 0 otherwise. Tri-state logic is not used with this operator.
         (The synonyms " .neqv." and " .xor." are also provided for compatibility with Fortran,
         although the second of these is not standard.)
    - `x1 .eqv. x2`: This is provided only for compatibility with Fortran and tests whether the boolean states
         of `x1` and `x2` (i.e. true/false) are equal. It is the negative of the exclusive OR (XOR) function.
         Tri-state logic is not used with this operator.
    - `!x`: Boolean unary NOT operation, returning 1 if `x` is zero, and 0 otherwise.
         (The synonym " .not." is also provided for compatibility with Fortran.)

    ### Relational Operators:

    Relational operators return the boolean result (0 or 1) of comparing the values of two floating point
    values for equality or inequality. The value AST__BAD may also be returned if either argument is `<bad>`.

    The following relational operators are available:
    - `x1 == x2`: Tests whether `x1` equals `x1`. (The synonym " .eq." is also provided
         for compatibility with Fortran.)
    - `x1 != x2`: Tests whether `x1` is unequal to `x2`.
        (The synonym " .ne." is also provided for compatibility with Fortran.)
    - `x1 > x2`: Tests whether `x1` is greater than `x2`.
        (The synonym " .gt." is also provided for compatibility with Fortran.)
    - `x1 >= x2`: Tests whether `x1` is greater than or equal to `x2`.
        (The synonym " .ge." is also provided for compatibility with Fortran.)
    - `x1 < x2`: Tests whether `x1` is less than `x2`.
        (The synonym " .lt." is also provided for compatibility with Fortran.)
    - `x1 <= x2`: Tests whether `x1` is less than or equal to `x2`.
        (The synonym " .le." is also provided for compatibility with Fortran.)

    Note that relational operators cannot usefully be used to compare values with the `<bad>` value
    (representing missing data), because the result is always `<bad>`. The isbad() function should be used
    instead.

    ### Bitwise Operators:

    The bitwise operators provided by C are often useful when operating on raw data (e.g.
    from instruments), so they are also provided for use in MathMap expressions.  In this
    case, however, the values on which they operate are floating point values rather than
    pure integers.  In order to produce results which match the pure integer case, the
    operands are regarded as fixed point binary numbers (i.e.  with the binary equivalent
    of a decimal point) with negative numbers represented using twos-complement notation.
    For integer values, the resulting bit pattern corresponds to that of the equivalent
    signed integer (digits to the right of the point being zero).  Operations on the bits
    representing the fractional part are also possible, however.

    The following bitwise operators are available:
    - `x1 >> x2`: Rightward bit shift. The integer value of `x2` is taken (rounding towards zero)
         and the bits representing `x1` are then shifted this number of places to the right
         (or to the left if the number of places is negative). This is equivalent to dividing
         `x1` by the corresponding power of 2.
    - `x1 << x2`: Leftward bit shift. The integer value of `x2` is taken (rounding towards zero),
         and the bits representing `x1` are then shifted this number of places to the left
         (or to the right if the number of places is negative). This is equivalent to
         multiplying `x1` by the corresponding power of 2.
    - `x1 & x2`: Bitwise AND between the bits of `x1` and those of `x2`
         (equivalent to a boolean AND applied at each bit position in turn).
    - `x1 | x2`: Bitwise OR between the bits of `x1` and those of `x2`
         (equivalent to a boolean OR applied at each bit position in turn).
    - `x1 ^ x2`: Bitwise exclusive OR (XOR) between the bits of `x1` and those of `x2`
         (equivalent to a boolean XOR applied at each bit position in turn).

    Note that no bit inversion operator (`~` in C) is provided. This is because inverting the bits
    of a twos-complement fixed point binary number is equivalent to simply negating it.
    This differs from the pure integer case because bits to the right of the binary point are also inverted.
    To invert only those bits to the left of the binary point, use a bitwise exclusive OR
    with the value -1 (i.e. `x^-1`).

    ### Functions:

    The following functions are available:
    - `abs(x)`: Absolute value of `x` (sign removal), same as fabs(x).
    - `acos(x)`: Inverse cosine of `x`, in radians.
    - `acosd(x)`: Inverse cosine of `x`, in degrees.
    - `acosh(x)`: Inverse hyperbolic cosine of `x`.
    - `acoth(x)`: Inverse hyperbolic cotangent of `x`.
    - `acsch(x)`: Inverse hyperbolic cosecant of `x`.
    - `aint(x)`: Integer part of `x` (round towards zero), same as `int(x)`.
    - `asech(x)`: Inverse hyperbolic secant of `x`.
    - `asin(x)`: Inverse sine of `x`, in radians.
    - `asind(x)`: Inverse sine of `x`, in degrees.
    - `asinh(x)`: Inverse hyperbolic sine of `x`.
    - `atan(x)`: Inverse tangent of `x`, in radians.
    - `atand(x)`: Inverse tangent of `x`, in degrees.
    - `atanh(x)`: Inverse hyperbolic tangent of `x`.
    - `atan2(x1, x2)`: Inverse tangent of `x1/x2`, in radians.
    - `atan2d(x1, x2)`: Inverse tangent of `x1/x2`, in degrees.
    - `ceil(x)`: Smallest integer value not less then `x` (round towards plus infinity).
    - `cos(x)`: Cosine of `x` in radians.
    - `cosd(x)`: Cosine of `x` in degrees.
    - `cosh(x)`: Hyperbolic cosine of `x`.
    - `coth(x)`: Hyperbolic cotangent of `x`.
    - `csch(x)`: Hyperbolic cosecant of `x`.
    - `dim(x1, x2)`: Returns `x1-x2` if `x1` is greater than `x2`, otherwise 0.
    - `exp(x)`: Exponential function of `x`.
    - `fabs(x)`: Absolute value of `x` (sign removal), same as abs(x).
    - `floor(x)`: Largest integer not greater than `x` (round towards minus infinity).
    - `fmod(x1, x2)`: Remainder when `x1` is divided by `x2`, same as mod(x1, x2).
    - `gauss(x1, x2)`: Random sample from a Gaussian distribution with mean `x1` and standard deviation `x2`.
    - `int(x)`: Integer part of `x` (round towards 0), same as `aint(x)`.
    - `isbad(x)`: Returns 1 if `x` has the
    - `log(x)`: Natural logarithm of `x`.
    - `log10(x)`: Logarithm of `x` to base
    - `max(x1, x2, ...)`: Maximum of two or
    - `min(x1, x2, ...)`: Minimum of two or
    - `mod(x1, x2)`: Remainder when `x1` is divided by `x2`, same as fmod(x1, x2).
    - `nint(x)`: Nearest integer to `x` (round to nearest).
    - `poisson(x)`: Random integer-valued sample from a Poisson distribution with mean `x`.
    - `pow(x1, x2)`: `x1` raised to the power of `x2`.
    - `qif(x1, x2, x3)`: Returns `x2` if `x1` is true, and " x3" otherwise.
    - `rand(x1, x2)`: Random sample from a uniform distribution in the range `x1` to `x2` inclusive.
    - `sech(x)`: Hyperbolic secant of `x`.
    - `sign(x1, x2)`: Absolute value of `x1` with the sign of `x2` (transfer of sign).
    - `sin(x)`: Sine of `x` in radians.
    - `sinc(x)`: Sinc function of `x` [= " sin(x)/x" ].
    - `sind(x)`: Sine of `x` in degrees.
    - `sinh(x)`: Hyperbolic sine of `x`.
    - `sqr(x)`: Square of `x` (= " x*x" ).
    - `sqrt(x)`: Square root of `x`.
    - `tan(x)`: Tangent of `x` in radians.
    - `tand(x)`: Tangent of `x` in degrees.
    - `tanh(x)`: Hyperbolic tangent of `x`.

    ### Symbolic Constants:

    The following symbolic constants are available (the enclosing `<>` brackets must be included):
    - `<bad>`: The "bad" value (AST__BAD) used to flag missing data. Note that you cannot
         usefully compare values with this constant because the result is always `<bad>`.
         The isbad() function should be used instead.
    - `<dig>`: Number of decimal digits of precision available in a floating point (double) value.
    - `<e>`: Base of natural logarithms.
    - `<epsilon>`: Smallest positive number such that `1.0+<epsilon>` is distinguishable from unity.
    - `<mant_dig>`: The number of base `<radix>` digits stored in the mantissa of a floating point (double)
        value.
    - `<max>`: Maximum representable floating point (double) value.
    - `<max_10_exp>`: Maximum integer such that 10 raised to that power can be represented
             as a floating point (double) value.
    - `<max_exp>`: Maximum integer such that `<radix>` raised to that power minus 1 can be represented
         as a floating point (double) value.
    - `<min>`: Smallest positive number which can be represented as a normalised floating point (double)
        value.
    - `<min_10_exp>`: Minimum negative integer such that 10 raised to that power can be represented as
         a normalised floating point (double) value.
    - `<min_exp>`: Minimum negative integer such that `<radix>` raised to that power minus 1
         can be represented as a normalised floating point (double) value.
    - `<pi>`: Ratio of the circumference of a circle to its diameter.
    - `<radix>`: The radix (number base) used to represent the mantissa of floating point (double) values.
    - `<rounds>`: The mode used for rounding floating point results after addition.
         Possible values include: -1 (indeterminate), 0 (toward zero), 1 (to nearest),
         2 (toward plus infinity) and 3 (toward minus infinity). Other values indicate
         machine-dependent behaviour.

    ### Evaluation Precedence and Associativity:

    Items appearing in expressions are evaluated in the following order (highest precedence first):
    - Constants and variables
    - Function arguments and parenthesised expressions - Function invocations
    - Unary `+ - ! .not.`
    - `**`
    - `*\/`
    - `+-`
    - `<< >>`
    - `< .lt. <=.le. > .gt. >=.ge.`
    - `== .eq. != .ne.`
    - `&`
    - `^`
    - `|`
    - `&& .and.`
    - `^^`
    - `|| .or.`
    - `.eqv. .neqv. .xor.`

    All operators associate from left-to-right, except for unary `+`, unary `-`, `!`, `.not.` and `**`
    which associate from right-to-left.
    */
    MathMap(int nin, int nout, std::vector<std::string> const &fwd, std::vector<std::string> const &rev,
            std::string const &options = "")
            : Mapping(reinterpret_cast<AstMapping *>(astMathMap(nin, nout, fwd.size(), getCStrVec(fwd).data(),
                                                                rev.size(), getCStrVec(rev).data(), "%s",
                                                                options.c_str()))) {
        assertOK();
    }

    virtual ~MathMap() {}

    /// Copy constructor: make a deep copy
    MathMap(MathMap const &) = default;
    MathMap(MathMap &&) = default;
    MathMap &operator=(MathMap const &) = delete;
    MathMap &operator=(MathMap &&) = default;

    /// Return a deep copy of this object.
    std::shared_ptr<MathMap> copy() const { return std::static_pointer_cast<MathMap>(copyPolymorphic()); }

    /**
    Get @ref MathMap_Seed "Seed": random number seed
    */
    int getSeed() const { return getI("Seed"); }

    /**
    Get @ref MathMap_SimpFI "SimpFI": can forward-inverse MathMap pairs safely simplify?
    */
    bool getSimpFI() const { return getB("SimpFI"); }

    /**
    Get @ref MathMap_SimpIF "SimpIF": can inverse-forward MathMap pairs safely simplify?
    */
    bool getSimpIF() const { return getB("SimpIF"); }

protected:
    virtual std::shared_ptr<Object> copyPolymorphic() const override {
        return copyImpl<MathMap, AstMathMap>();
    }

    /// Construct a MathMap from a raw AST pointer
    explicit MathMap(AstMathMap *rawptr) : Mapping(reinterpret_cast<AstMapping *>(rawptr)) {
        if (!astIsAMathMap(getRawPtr())) {
            std::ostringstream os;
            os << "this is a " << getClassName() << ", which is not a MathMap";
            throw std::invalid_argument(os.str());
        }
    }

private:
    /// Convert a vector<string> to a vector<char const *>
    std::vector<char const *> getCStrVec(std::vector<std::string> const &strVec) {
        std::vector<char const *> cstrVec;
        for (auto const &str : strVec) {
            cstrVec.push_back(str.c_str());
        }
        return cstrVec;
    }
};

}  // namespace ast

#endif
