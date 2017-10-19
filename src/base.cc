#include "astshim/base.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ast {
namespace {

static std::ostringstream errorMsgStream;

/*
Write an error message to `errorMsgStream`

Intended to be registered as an error handler to AST by calling `astSetPutErr(reportError)`.
*/
void reportError(int errNum, const char *errMsg) { errorMsgStream << errMsg; }

/*
Instantiate this class to register `reportError` as an AST error handler.
*/
class ErrorHandler {
public:
    ErrorHandler() { astSetPutErr(reportError); }

    ErrorHandler(ErrorHandler const &) = delete;
    ErrorHandler(ErrorHandler &&) = delete;
    ErrorHandler &operator=(ErrorHandler const &) = delete;
    ErrorHandler &operator=(ErrorHandler &&) = delete;

    static std::string getErrMsg() {
        auto errMsg = errorMsgStream.str();
        // clear status bits
        errorMsgStream.clear();
        if (errMsg.empty()) {
            errMsg = "Failed with AST status = " + std::to_string(astStatus);
        } else {
            // empty the stream
            errorMsgStream.str("");
        }
        astClearStatus;
        return errMsg;
    }
};

}  // namespace

void assertOK(AstObject *rawPtr1, AstObject *rawPtr2) {
    // Construct ErrorHandler once, the first time this function is called.
    // This is done to initialize `errorMsgStream` and register `reportError` as the AST error handler.
    // See https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
    static ErrorHandler *errHandler = new ErrorHandler();
    if (!astOK) {
        if (rawPtr1) {
            astAnnul(rawPtr1);
        }
        if (rawPtr2) {
            astAnnul(rawPtr2);
        }
        throw std::runtime_error(errHandler->getErrMsg());
    }
}

ConstArray2D arrayFromVector(std::vector<double> const &vec, int nAxes) {
    return static_cast<ConstArray2D>(arrayFromVector(const_cast<std::vector<double> &>(vec), nAxes));
}

Array2D arrayFromVector(std::vector<double> &vec, int nAxes) {
    int nPoints = vec.size() / nAxes;
    if (nPoints * nAxes != vec.size()) {
        std::ostringstream os;
        os << "vec length = " << vec.size() << " not a multiple of nAxes = " << nAxes;
        throw std::runtime_error(os.str());
    }
    Array2D::Index shape = ndarray::makeVector(nAxes, nPoints);
    Array2D::Index strides = ndarray::makeVector(nPoints, 1);
    return external(vec.data(), shape, strides);
}

}  // namespace ast