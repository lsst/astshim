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
#include <iostream>
#include <memory>

#include "astshim.h"

typedef ndarray::Array<double, 2, 2> PointArray;

PointArray arrayFromVec(std::vector<double> &vec, int width=2) {
    if (vec.size() % width != 0) {
        std::ostringstream os;
        os << "vec length = " << vec.size() << " not a multiple of width = " << width;
        throw std::runtime_error(os.str());
    }
    int nPoints = vec.size() / width;
    PointArray::Index shape = ndarray::makeVector(nPoints, width);
    PointArray::Index strides = ndarray::makeVector(width, 1);
    return external(vec.data(), shape, strides);
}

int main() {
    auto stream = ast::FileStream("examples/simple.fits");
    auto channel = ast::FitsChan(stream);
    auto frameSetPtr = std::dynamic_pointer_cast<ast::FrameSet>(channel.read());

    auto baseFrame = frameSetPtr->getFrame(ast::FrameSet::BASE);
    std::cout << "base (input) domain = " << baseFrame->getDomain() << std::endl;
    auto currentFrame = frameSetPtr->getFrame(ast::FrameSet::CURRENT);
    std::cout << "current (output) domain = " << currentFrame->getDomain() << std::endl;

    // transform some points
    std::vector<double> fromVec = {0, 0, 1000, 0, 0, 1000, 1000, 1000};
    PointArray from = arrayFromVec(fromVec);
    PointArray to = ndarray::allocate(ndarray::makeVector(4, 2));
    frameSetPtr->tran(from, to);
    std::cout << "\n\npixels =" << from << std::endl;
    std::cout << "sky =" << to << std::endl;

}
