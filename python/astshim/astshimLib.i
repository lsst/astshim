// -*- lsst-c++ -*-
%define daf_base_DOCSTRING
"
A C++ shim around a subset of AST
"
%enddef

%feature("autodoc", "1");
%module(package="astshim", docstring=daf_base_DOCSTRING) astshimLib

%{
#include <complex>
#include <memory>
#include <string>
#include "astshim.h"
%}

%include "lsst/p_lsstSwig.i"  // for initializeNumPy; defined in LSST package "utils"
%initializeNumPy(astshim)

%include "exception.i"
%exception {
    try {
        $action
    } catch (std::exception & e) {
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
}

%{
#include "ndarray/swig.h"
#include "ndarray/converter/eigen.h"
%}

%include "ndarray.i"

%declareNumPyConverters(ndarray::Array<double, 2, 2>);

%include "std_vector.i"
%template(VectorDouble) std::vector<double>;
%template(VectorInt) std::vector<int>;
%template(VectorString) std::vector<std::string>;

%include "std_complex.i"

%include "std_shared_ptr.i"
%shared_ptr(ast::Object)
%shared_ptr(ast::Stream);
%shared_ptr(ast::Mapping)
%shared_ptr(ast::Channel)

// mappings
%shared_ptr(ast::CmpMap)
%shared_ptr(ast::ParallelMap)
%shared_ptr(ast::SeriesMap)
%shared_ptr(ast::ZoomMap)

%shared_ptr(ast::FileStream);
%shared_ptr(ast::StringStream);

%include "astshim/base.h"
%include "astshim/Object.h"
%include "astshim/Stream.h"
%include "astshim/Channel.h"
%include "astshim/Mapping.h"

%include "astshim/CmpMap.h"
%include "astshim/ParallelMap.h"
%include "astshim/SeriesMap.h"
%include "astshim/ZoomMap.h"

%define %addRepr(CLS...)
%extend ast::CLS {
    std::string __repr__() const {
        return self->show();
    }
    std::string __str__() const {
        return self->getClass();
    }
}
%enddef

%addRepr(Object)
// %addRepr(Stream)  // needs special treatment
%addRepr(Channel)
%addRepr(Mapping)

%addRepr(CmpMap)
%addRepr(ParallelMap)
%addRepr(SeriesMap)
%addRepr(ZoomMap)


