import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.coord import IcrsCoord


def makeSimpleImage():
    wcs = afwGeom.makeSkyWcs(
        crpix = afwGeom.Point2D(0, 0),
        crval = IcrsCoord(0 * afwGeom.degrees, 0 * afwGeom.degrees),
        cdMatrix = afwGeom.makeCdMatrix(scale = 2 * afwGeom.arcseconds),
    )
    im = afwImage.ExposureF(100, 100, wcs)
    im.writeFits("simple.fits")


if __name__ == "__main__":
    makeSimpleImage()
