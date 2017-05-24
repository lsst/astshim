from __future__ import absolute_import, division, print_function
import os.path
import unittest

import astshim
from astshim.test import ObjectTestCase

DataDir = os.path.join(os.path.dirname(__file__))


def pad(card):
    """Pad a string withs paces to length 80 characters"""
    return "%-80s" % (card,)


class TestObject(ObjectTestCase):

    def setUp(self):
        shortCards = (
            "NAXIS1  =                  200",
            "NAXIS2  =                  200",
            "CTYPE1  = 'RA--TAN '",
            "CTYPE2  = 'DEC-TAN '",
            "CRPIX1  =                  100",
            "CRPIX2  =                  100",
            "CDELT1  =                0.001",
            "CDELT2  =                0.001",
            "CRVAL1  =                    0",
            "CRVAL2  =                    0",
        )
        self.cards = [pad(card) for card in shortCards]

    def test_FitsChanPreloaded(self):
        """Test a FitsChan that starts out loaded with data
        """
        ss = astshim.StringStream("".join(self.cards))
        fc = astshim.FitsChan(ss)
        self.assertEqual(fc.getNCard(), len(self.cards))
        self.assertEqual(fc.getClassName(), "FitsChan")
        fv = fc.getFitsF("CRVAL1")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, 0.0)

        self.assertEqual(fc.getEncoding(), "FITS-WCS")

    def test_FitsChanFileStream(self):
        """Test a FitsChan with a FileStream

        In particular, make sure that cards are written as the channel is destroyed
        """
        path = os.path.join(DataDir, "test_fitsChanFileStream.fits")
        fc1 = astshim.FitsChan(astshim.FileStream(path, True))
        fc1.putCards("".join(self.cards))
        # delete the channel, which writes cards,
        # and then deletes the file stream, closing the file
        del fc1

        fc2 = astshim.FitsChan(astshim.FileStream(path, False))
        self.assertEqual(fc2.getNCard(), len(self.cards))
        del fc2
        os.remove(path)

    def test_FitsChanWriteOnDelete(self):
        """Test that a FitsChan writes cards when it is deleted
        """
        ss = astshim.StringStream()
        fc = astshim.FitsChan(ss)
        fc.putCards("".join(self.cards))
        self.assertEqual(ss.getSinkData(), "")
        del fc
        self.assertEqual(len(ss.getSinkData()), 80 * len(self.cards))

    def test_FitsChanGetFits(self):
        fc = astshim.FitsChan(astshim.StringStream())
        self.assertEqual(fc.getClassName(), "FitsChan")
        fc.setFitsI("FRED", 99, "Hello there", True)
        fv = fc.getFitsI("FRED")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, 99)
        fv = fc.getFitsS("FRED")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, "99")
        self.assertEqual(fc.getNCard(), 1)
        self.assertEqual(fc.getAllCardNames(), ["FRED"])
        # replace this card
        fc.setFitsF("FRED1", 99.9, "Hello there", True)
        fv = fc.getFitsS("FRED1")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, "99.9")
        self.assertEqual(fc.getNCard(), 1)
        self.assertEqual(fc.getAllCardNames(), ["FRED1"])
        fc.setFitsCF("FRED1", complex(99.9, 99.8), "Hello there", True)
        fv = fc.getFitsCF("FRED1")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value.real, 99.9)
        self.assertEqual(fv.value.imag, 99.8)
        fc.setFitsS("FRED1", "-12", "Hello there", True)
        fv = fc.getFitsI("FRED1")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, -12)
        self.assertEqual(fc.getNCard(), 1)
        self.assertEqual(fc.getAllCardNames(), ["FRED1"])

    def test_FitsChanEmptyFits(self):
        ss = astshim.StringStream("".join(self.cards))
        fc = astshim.FitsChan(ss)
        self.assertEqual(fc.getNCard(), len(self.cards))
        fc.emptyFits()
        self.assertEqual(fc.getNCard(), 0)

    def test_FitsChanPutCardsPutFits(self):
        ss = astshim.StringStream()
        fc = astshim.FitsChan(ss)
        cards = "CRVAL1  = 0                                                                     " + \
                "CRVAL2  = 0                                                                     "
        fc.putCards(cards)
        self.assertEqual(fc.getCard(), 1)
        fc.setCard(100)  # past the end = end of cards
        self.assertEqual(fc.getCard(), 3)
        fc.clearCard()
        self.assertEqual(fc.getCard(), 1)
        self.assertEqual(fc.getAllCardNames(), ["CRVAL1", "CRVAL2"])

        # insert new cards at the beginning
        for card in self.cards[0:8]:
            fc.putFits(card, False)
        self.assertEqual(fc.getNCard(), 10)
        self.assertEqual(fc.getCard(), 9)
        predCardNames = [c.split()[0] for c in self.cards[0:8]] + ["CRVAL1", "CRVAL2"]
        self.assertEqual(fc.getAllCardNames(), predCardNames)

    def test_FitsChanFindFits(self):
        ss = astshim.StringStream("".join(self.cards))
        fc = astshim.FitsChan(ss)
        fc.setCard(9)
        fv = fc.findFits("%f", False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                    0"))
        fc.delFits()
        self.assertEqual(fc.getNCard(), len(self.cards) - 1)
        self.assertEqual(fc.getCard(), 9)
        fv = fc.findFits("%f", False)
        self.assertEqual(fc.getCard(), 9)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                    0"))
        fc.putFits("CRVAL1  = 0", False)
        self.assertEqual(fc.getNCard(), len(self.cards))
        self.assertEqual(fc.getCard(), 10)
        fv = fc.findFits("%f", False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                    0"))

        fv = fc.findFits("CTYPE2", False)
        self.assertFalse(fv.found)

        fc.clearCard()
        fv = fc.findFits("CTYPE2", False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE2  = 'DEC-TAN '"))
        self.assertEqual(fc.getCard(), 4)

    def test_FitsChanReadWrite(self):
        ss = astshim.StringStream("".join(self.cards))
        fc1 = astshim.FitsChan(ss)
        obj1 = fc1.read()
        self.assertEqual(obj1.getClassName(), "FrameSet")

        ss2 = astshim.StringStream()
        fc2 = astshim.FitsChan(ss2, "Encoding=FITS-WCS")
        n = fc2.write(obj1)
        self.assertEqual(n, 1)
        self.assertEqual(fc2.getNCard(), 10)
        fc2.clearCard()

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("WCSAXES =                    2 / Number of WCS axes"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRPIX1  =                100.0 / Reference pixel on axis 1"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRPIX2  =                100.0 / Reference pixel on axis 2"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                  0.0 / Value at ref. pixel on axis 1"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                  0.0 / Value at ref. pixel on axis 2"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CDELT1  =                0.001 / Pixel size on axis 1"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CDELT2  =                0.001 / Pixel size on axis 2"))

        fv = fc2.findFits("%f", True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("RADESYS = 'ICRS    '           / Reference frame for RA/DEC values"))

        self.assertEqual(ss2.getSinkData(), "")
        self.assertEqual(fc2.getNCard(), 10)
        fc2.writeFits()
        self.assertEqual(fc2.getNCard(), 0)
        a = ss2.getSinkData()

        ss3 = astshim.StringStream(ss2.getSinkData())
        fc3 = astshim.FitsChan(ss3, "Encoding=FITS-WCS")
        fc3.readFits()
        obj3 = fc3.read()

        ss4 = astshim.StringStream()
        fc4 = astshim.FitsChan(ss4, "Encoding=FITS-WCS")
        n = fc4.write(obj3)
        self.assertEqual(n, 1)
        del fc4
        b = ss4.getSinkData()
        self.assertEqual(a, b)

    def test_FitsChanGetFitsSetFits(self):
        fc = astshim.FitsChan(astshim.StringStream())
        fc.setFitsI("NAXIS1", 200)
        fc.setFitsI("NAXIS2", 200)
        fc.setFitsS("CTYPE1", "RA--TAN")
        fc.setFitsS("CTYPE2", "DEC-TAN")
        fc.setFitsI("CRPIX1", 100)
        fc.setFitsI("CRPIX2", 100)
        fc.setFitsF("CDELT1", 0.001)
        fc.setFitsF("CDELT2", 0.001)
        fc.setFitsI("CRVAL1", 0)
        fc.setFitsI("CRVAL2", 0)
        fc.setFitsCF("ACPLX", complex(-5.5, 4.3))
        fc.setFitsL("ABOOL", False)
        fc.setFitsCN("ACN", "continue_value")
        fc.setFitsU("ANUNK")

        self.assertEqual(fc.getNCard(), 14)

        fv = fc.getFitsI("NAXIS1", 0)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, 200)
        fv = fc.getFitsS("CTYPE1", "")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, "RA--TAN")
        fv = fc.getFitsF("CDELT2", -53.2)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, 0.001)
        fv = fc.getFitsCF("ACPLX")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, complex(-5.5, 4.3))
        fv = fc.getFitsCN("ACN")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, "continue_value")

        # test getFitsX for missing cards with default values
        fv = fc.getFitsCF("BOGUS", complex(9, -5))
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, complex(9, -5))
        fv = fc.getFitsCN("BOGUS", "not_there")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "not_there")
        fv = fc.getFitsF("BOGUS", 55.5)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 55.5)
        fv = fc.getFitsI("BOGUS", 55)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 55)
        fv = fc.getFitsL("BOGUS", True)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, True)
        fv = fc.getFitsS("BOGUS", "missing")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "missing")

        # test getFitsX for missing cards without default values
        fv = fc.getFitsCF("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, complex())
        fv = fc.getFitsCN("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")
        fv = fc.getFitsF("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 0)
        fv = fc.getFitsI("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 0)
        fv = fc.getFitsL("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, False)
        fv = fc.getFitsS("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")

        # the only way to find a card with unknown value is with findFits
        fc.clearCard()
        fv = fc.findFits("ANUNK", False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value,
                         'ANUNK   =                                                                       ')

        fv = fc.findFits("BOGUS", False)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")


if __name__ == "__main__":
    unittest.main()
