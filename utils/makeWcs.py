import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage


def makeSimpleImage():
    wcs = afwGeom.makeSkyWcs(
        crpix = afwGeom.Point2D(0, 0),
        crval = afwGeom.SpherePoint(0, 0, afwGeom.degrees),
        cdMatrix = afwGeom.makeCdMatrix(scale = 2 * afwGeom.arcseconds),
    )
    im = afwImage.ExposureF(100, 100, wcs)
    im.writeFits("simple.fits")


if __name__ == "__main__":
    makeSimpleImage()
