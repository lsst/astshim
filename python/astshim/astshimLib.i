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
#include "astshim/makeBadMatrixMap.h"
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
%shared_ptr(ast::Frame)
%shared_ptr(ast::FrameSet)

// channels
%shared_ptr(ast::FitsChan)
%shared_ptr(ast::XmlChan)

// frames
%shared_ptr(ast::CmpFrame)
%shared_ptr(ast::SkyFrame)
%shared_ptr(ast::SpecFrame)
%shared_ptr(ast::TimeFrame)

// mappings
%shared_ptr(ast::CmpMap)
%shared_ptr(ast::LutMap)
%shared_ptr(ast::MathMap)
%shared_ptr(ast::makeBadMatrixMap)
%shared_ptr(ast::MatrixMap)
%shared_ptr(ast::NormMap)
%shared_ptr(ast::ParallelMap)
%shared_ptr(ast::PcdMap)
%shared_ptr(ast::PermMap)
%shared_ptr(ast::PolyMap)
%shared_ptr(ast::RateMap)
%shared_ptr(ast::SeriesMap)
%shared_ptr(ast::ShiftMap)
%shared_ptr(ast::SlaMap)
%shared_ptr(ast::SphMap)
%shared_ptr(ast::TimeMap)
%shared_ptr(ast::TranMap)
%shared_ptr(ast::UnitMap)
%shared_ptr(ast::UnitNormMap)
%shared_ptr(ast::WcsMap)
%shared_ptr(ast::WinMap)
%shared_ptr(ast::ZoomMap)

%shared_ptr(ast::FileStream);
%shared_ptr(ast::StringStream);

%include "astshim/base.h"
%include "astshim/Object.h"
%include "astshim/Stream.h"
%include "astshim/Channel.h"
%include "astshim/MapBox.h"
%include "astshim/MapSplit.h"
%include "astshim/QuadApprox.h"
%include "astshim/Mapping.h"
%include "astshim/Frame.h"
%include "astshim/FrameSet.h"

// channels
%include "astshim/FitsChan.h"
%include "astshim/XmlChan.h"

// templates needed for FitsChan's FoundValue
%template(FoundValueBool) ast::FoundValue<bool>;
%template(FoundValueComplex) ast::FoundValue<std::complex<double>>;
%template(FoundValueDouble) ast::FoundValue<double>;
%template(FoundValueInt) ast::FoundValue<int>;
%template(FoundValueString) ast::FoundValue<std::string>;

// frames
%include "astshim/CmpFrame.h"
%include "astshim/SkyFrame.h"
%include "astshim/SpecFrame.h"
%include "astshim/TimeFrame.h"

// mappings
%include "astshim/CmpMap.h"
%include "astshim/LutMap.h"
%include "astshim/MathMap.h"
%include "astshim/makeBadMatrixMap.h"
%include "astshim/MatrixMap.h"
%include "astshim/NormMap.h"
%include "astshim/PcdMap.h"
%include "astshim/ParallelMap.h"
%include "astshim/PermMap.h"
%include "astshim/PolyMap.h"
%include "astshim/RateMap.h"
%include "astshim/SeriesMap.h"
%include "astshim/ShiftMap.h"
%include "astshim/SlaMap.h"
%include "astshim/SphMap.h"
%include "astshim/TimeMap.h"
%include "astshim/TranMap.h"
%include "astshim/UnitMap.h"
%include "astshim/UnitNormMap.h"
%include "astshim/WcsMap.h"
%include "astshim/WinMap.h"
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
%addRepr(Frame)
%addRepr(FrameSet)

// frames
%addRepr(CmpFrame)
%addRepr(SkyFrame)
%addRepr(SpecFrame)
%addRepr(TimeFrame)

// mappings
%addRepr(CmpMap)
%addRepr(LutMap)
%addRepr(MathMap)
%addRepr(MatrixMap)
%addRepr(NormMap)
%addRepr(ParallelMap)
%addRepr(PcdMap)
%addRepr(PermMap)
%addRepr(PolyMap)
%addRepr(RateMap)
%addRepr(SeriesMap)
%addRepr(ShiftMap)
%addRepr(SlaMap)
%addRepr(SphMap)
%addRepr(TimeMap)
%addRepr(TranMap)
%addRepr(UnitMap)
%addRepr(UnitNormMap)
%addRepr(WcsMap)
%addRepr(WinMap)
%addRepr(ZoomMap)


