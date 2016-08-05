#include "astshim/base.h"

namespace ast {

ConstArray2D arrayFromVector(std::vector<double> const &vec, int nAxes) {
    if (vec.size() % nAxes != 0) {
        std::ostringstream os;
        os << "vec length = " << vec.size() << " not a multiple of nAxes = " << nAxes;
        throw std::runtime_error(os.str());
    }
    int nPoints = vec.size() / nAxes;
    Array2D::Index shape = ndarray::makeVector(nPoints, nAxes);
    Array2D::Index strides = ndarray::makeVector(nAxes, 1);
    return external(vec.data(), shape, strides);
}

Array2D arrayFromVector(std::vector<double> &vec, int nAxes) {
    if (vec.size() % nAxes != 0) {
        std::ostringstream os;
        os << "vec length = " << vec.size() << " not a multiple of nAxes = " << nAxes;
        throw std::runtime_error(os.str());
    }
    int nPoints = vec.size() / nAxes;
    Array2D::Index shape = ndarray::makeVector(nPoints, nAxes);
    Array2D::Index strides = ndarray::makeVector(nAxes, 1);
    return external(vec.data(), shape, strides);
}

} // ast