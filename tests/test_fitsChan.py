from __future__ import absolute_import, division, print_function
import os.path
import unittest
import numpy as np

import astshim as ast
from astshim.test import ObjectTestCase


def pad(card):
    """Pad a string withs paces to length 80 characters"""
    return "%-80s" % (card,)


def writeFitsWcs(frameSet, extraOptions=None):
    """Write a FrameSet as FITS-WCS

    extraOptions are in addition to Encoding=Fits-WCS, CDMatrix=1
    """
    options = "Encoding=FITS-WCS, CDMatrix=1"
    if extraOptions is not None:
        options = "%s, %s" % (options, extraOptions)
    fc = ast.FitsChan(ast.StringStream(), options)
    fc.write(frameSet)
    return fc


class TestFitsChan(ObjectTestCase):

    def setUp(self):
        self.dataDir = os.path.join(os.path.dirname(__file__), "data")
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
            "BOOL    =                    F",
            "UNDEF   =",
            "BOOL    =                    T / Repeat",
            "COMMENT  one of two comments",
            "COMMENT  another of two comments",
            "HISTORY  one of two history fields",
            "HISTORY  second of three history fields",
            "HISTORY  third of three history fields",
        )
        self.cards = [pad(card) for card in shortCards]

    def insertPixelMapping(self, mapping, frameSet):
        """Make a new WCS by inserting a new mapping at the beginnning of the GRID-IWC mapping

        Return the new FrameSet (the original is not altered).
        """
        frameSet = frameSet.copy()

        skyFrame = frameSet.getFrame(ast.FrameSet.CURRENT)  # use this copy for the new sky frame
        self.assertIsInstance(skyFrame, ast.SkyFrame)
        oldSkyIndex = frameSet.current

        if not frameSet.findFrame(ast.Frame(2, "Domain=GRID")):
            raise KeyError("No GRID frame")
        gridIndex = frameSet.current

        if not frameSet.findFrame(ast.Frame(2, "Domain=IWC")):
            raise KeyError("No IWC frame")
        oldIwcIndex = frameSet.current
        iwcFrame = frameSet.getFrame(oldIwcIndex)  # use this copy for the new IWC frame

        oldGridToIwc = frameSet.getMapping(gridIndex, oldIwcIndex)
        iwcToSky = frameSet.getMapping(oldIwcIndex, oldSkyIndex)

        # Remove frames in order high to low, so removal doesn't alter the indices remaining to be removed;
        # update gridIndex during removal so it still points to the GRID frame
        framesToRemove = reversed(sorted([oldIwcIndex, oldSkyIndex]))
        for index in framesToRemove:
            if (index < gridIndex):
                gridIndex -= 1
            frameSet.removeFrame(index)

        newGridToIwc = mapping.then(oldGridToIwc).simplified()
        frameSet.addFrame(gridIndex, newGridToIwc, iwcFrame)
        frameSet.addFrame(ast.FrameSet.CURRENT, iwcToSky, skyFrame)
        return frameSet

    def test_FitsChanAttributes(self):
        """Test getting and setting FitsChan attributes

        Does not test the behavior of the attributes.
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        self.assertFalse(fc.carLin)
        self.assertFalse(fc.cdMatrix)
        self.assertFalse(fc.clean)
        self.assertFalse(fc.defB1950)
        self.assertEqual(fc.encoding, "FITS-WCS")
        self.assertEqual(fc.fitsAxisOrder, "<auto>")
        self.assertAlmostEqual(fc.fitsTol, 0.1)
        self.assertFalse(fc.iwc)
        self.assertTrue(fc.sipOK)
        self.assertTrue(fc.sipReplace)
        self.assertEqual(fc.tabOK, 0)
        self.assertEqual(fc.polyTan, -1)
        warningSet = set(fc.warnings.split(" "))
        desiredWarningSet = set("BadKeyName BadKeyValue Tnx Zpx BadCel BadMat BadPV BadCTYPE".split(" "))
        self.assertEqual(warningSet, desiredWarningSet)

        fc.carLin = True
        self.assertTrue(fc.carLin)
        fc.cdMatrix = True
        self.assertTrue(fc.cdMatrix)
        fc.clean = True
        self.assertTrue(fc.clean)
        fc.defB1950 = True
        self.assertTrue(fc.defB1950)
        fc.encoding = "NATIVE"
        self.assertEqual(fc.encoding, "NATIVE")
        fc.fitsAxisOrder = "<copy>"
        self.assertEqual(fc.fitsAxisOrder, "<copy>")
        fc.fitsTol = 0.001
        self.assertAlmostEqual(fc.fitsTol, 0.001)
        fc.iwc = True
        self.assertTrue(fc.iwc)
        fc.tabOK = 1
        self.assertEqual(fc.tabOK, 1)
        fc.polyTan = 0
        self.assertEqual(fc.polyTan, 0)
        fc.warnings = "BadKeyName BadMat"
        self.assertEqual(fc.warnings, "BadKeyName BadMat")

    def test_FitsChanPreloaded(self):
        """Test a FitsChan that starts out loaded with data
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        self.assertEqual(fc.nCard, len(self.cards))
        # there are 2 COMMENT and 3 HISTORY cards,
        # and two BOOL cards so 4 fewer unique keys
        self.assertEqual(fc.nKey, len(self.cards) - 4)
        self.assertEqual(fc.className, "FitsChan")
        fv = fc.getFitsF("CRVAL1")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, 0.0)

        self.assertEqual(fc.encoding, "FITS-WCS")

        self.assertEqual(fc.getAllCardNames(),
                         [card.split(" ", 1)[0] for card in self.cards])

    def test_FitsChanFileStream(self):
        """Test a FitsChan with a FileStream

        In particular, make sure that cards are written as the channel is destroyed
        """
        path = os.path.join(self.dataDir, "test_fitsChanFileStream.fits")
        fc1 = ast.FitsChan(ast.FileStream(path, True))
        fc1.putCards("".join(self.cards))
        # delete the channel, which writes cards,
        # and then deletes the file stream, closing the file
        del fc1

        fc2 = ast.FitsChan(ast.FileStream(path, False))
        self.assertEqual(fc2.nCard, len(self.cards))
        del fc2
        os.remove(path)

    def test_FitsChanWriteOnDelete(self):
        """Test that a FitsChan writes cards when it is deleted
        """
        ss = ast.StringStream()
        fc = ast.FitsChan(ss)
        fc.putCards("".join(self.cards))
        self.assertEqual(ss.getSinkData(), "")
        del fc
        self.assertEqual(len(ss.getSinkData()), 80 * len(self.cards))

    def test_FitsChanGetFitsSetFits(self):
        """Test FitsChan.getFits<X>, FitsChan.setFits<X> and getCardType
        """
        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")

        # add a card for each type
        complexVal = complex(9.8, -5.15)
        continueVal = "This is a continue card"
        floatVal = 1.5
        intVal = 99
        logicalVal = True
        strVal = "This is a string"
        fc.setFitsCF("ACOMPLEX", complexVal, "Comment for ACOMPLEX")
        commentVal = "This is a comment"
        fc.setFitsCN("ACONT", continueVal, "Comment for ACONT")
        fc.setFitsF("AFLOAT", floatVal, "Comment for AFLOAT")
        fc.setFitsI("ANINT", intVal, "Comment for ANINT")
        fc.setFitsL("ALOGICAL", logicalVal, "Comment for ALOGICAL")
        fc.setFitsS("ASTRING", strVal, "Comment for ASTRING")
        fc.setFitsU("UNDEFVAL", "Comment for UNDEFVAL")
        fc.setFitsCM(commentVal)

        self.assertEqual(fc.nCard, 8)
        self.assertEqual(fc.getAllCardNames(),
                         ["ACOMPLEX", "ACONT", "AFLOAT", "ANINT",
                          "ALOGICAL", "ASTRING", "UNDEFVAL", "        "])

        fv = fc.getFitsI("ANINT")
        self.assertEqual(fc.getCardType(), ast.CardType.INT)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, intVal)
        self.assertEqual(fc.getCardComm(), "Comment for ANINT")
        self.assertEqual(fc.getCard(), 4)

        fv = fc.getFitsS("ANINT")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, str(intVal))
        self.assertEqual(fc.getCard(), 4)

        fv = fc.getFitsI()  # read the current card
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, intVal)
        self.assertEqual(fc.getCard(), 4)

        fv = fc.getFitsI("")  # alternate way to read the current card
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, intVal)
        self.assertEqual(fc.getCard(), 4)

        fv = fc.getFitsF("AFLOAT")
        self.assertEqual(fc.getCardType(), ast.CardType.FLOAT)
        self.assertEqual(fc.getCard(), 3)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, floatVal)
        self.assertEqual(fc.getCardComm(), "Comment for AFLOAT")

        fv = fc.getFitsF()  # read the current card
        self.assertEqual(fc.getCardType(), ast.CardType.FLOAT)
        self.assertEqual(fc.getCard(), 3)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, floatVal)
        self.assertEqual(fc.getCardComm(), "Comment for AFLOAT")

        fv = fc.getFitsCN("ACONT")
        self.assertEqual(fc.getCardType(), ast.CardType.CONTINUE)
        self.assertEqual(fc.getCard(), 2)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, continueVal)
        self.assertEqual(fc.getCardComm(), "Comment for ACONT")

        fv = fc.getFitsCN()  # read the current card
        self.assertEqual(fc.getCardType(), ast.CardType.CONTINUE)
        self.assertEqual(fc.getCard(), 2)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, continueVal)
        self.assertEqual(fc.getCardComm(), "Comment for ACONT")

        fv = fc.getFitsCF("ACOMPLEX")
        self.assertEqual(fc.getCardType(), ast.CardType.COMPLEXF)
        self.assertEqual(fc.getCard(), 1)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, complexVal)
        self.assertEqual(fc.getCardComm(), "Comment for ACOMPLEX")

        fv = fc.getFitsCF()  # read the current card
        self.assertEqual(fc.getCardType(), ast.CardType.COMPLEXF)
        self.assertEqual(fc.getCard(), 1)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, complexVal)
        self.assertEqual(fc.getCardComm(), "Comment for ACOMPLEX")

        fv = fc.getFitsL("ALOGICAL")
        self.assertEqual(fc.getCardType(), ast.CardType.LOGICAL)
        self.assertEqual(fc.getCard(), 5)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, logicalVal)
        self.assertEqual(fc.getCardComm(), "Comment for ALOGICAL")

        fv = fc.getFitsL()  # read the current card
        self.assertEqual(fc.getCardType(), ast.CardType.LOGICAL)
        self.assertEqual(fc.getCard(), 5)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, logicalVal)
        self.assertEqual(fc.getCardComm(), "Comment for ALOGICAL")

        fv = fc.getFitsS("ASTRING")
        self.assertEqual(fc.getCardType(), ast.CardType.STRING)
        self.assertEqual(fc.getCard(), 6)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, strVal)
        self.assertEqual(fc.getCardComm(), "Comment for ASTRING")

        fv = fc.getFitsS()  # read the current card
        self.assertEqual(fc.getCardType(), ast.CardType.STRING)
        self.assertEqual(fc.getCard(), 6)
        self.assertTrue(fv.found)
        self.assertAlmostEqual(fv.value, strVal)
        self.assertEqual(fc.getCardComm(), "Comment for ASTRING")

        fv = fc.getFitsS("BADNAME")  # a card that does not exist
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)
        self.assertFalse(fv.found)

        fc.setCard(7)
        self.assertEqual(fc.getCardType(), ast.CardType.UNDEF)
        with self.assertRaises(RuntimeError):
            fc.getFitsS()
        self.assertEqual(fc.getCardComm(), "Comment for UNDEFVAL")

        fc.setCard(8)
        self.assertEqual(fc.getCardType(), ast.CardType.COMMENT)
        with self.assertRaises(RuntimeError):
            fc.getFitsS()
        self.assertEqual(fc.getCardComm(), commentVal)

        # replace ANINT card with new everything: name, value type and comment
        fc.setCard(4)
        fc.setFitsCF("NEWNAME", complex(99.9, 99.8), "New comment", overwrite=True)
        self.assertEqual(fc.getAllCardNames(),
                         ["ACOMPLEX", "ACONT", "AFLOAT", "NEWNAME",
                          "ALOGICAL", "ASTRING", "UNDEFVAL", "        "])
        fc.setCard(1)  # force a search
        fv = fc.getFitsCF("NEWNAME")
        self.assertTrue(fv.found)
        self.assertEqual(fv.value.real, 99.9)
        self.assertEqual(fv.value.imag, 99.8)
        self.assertEqual(fc.getCardComm(), "New comment")
        self.assertEqual(fc.getCard(), 4)

    def test_FitsChanGetCurrentForNonexistentCard(self):
        """Test getting info on the current card when it does not exist
        """
        fc = ast.FitsChan(ast.StringStream())
        fc.setFitsI("ANINT", 200)
        fc.setFitsS("ASTRING", "string value")
        fc.setCard(fc.nCard + 1)
        self.assertEqual(fc.getCardType(), ast.NOTYPE)
        self.assertEqual(fc.testFits(), ast.ABSENT)
        with self.assertRaises(RuntimeError):
            fc.getFitsCN()
        with self.assertRaises(RuntimeError):
            fc.getFitsF()
        with self.assertRaises(RuntimeError):
            fc.getFitsI()
        with self.assertRaises(RuntimeError):
            fc.getFitsL()
        with self.assertRaises(RuntimeError):
            fc.getFitsS()
        with self.assertRaises(RuntimeError):
            fc.getFitsCF()
        self.assertEqual(fc.getCardName(), "")
        self.assertEqual(fc.getCardComm(), "")

    def test_FitsChanGetFitsMissing(self):
        """Test FitsChan.getFits<X> for missing cards, with and without defaults
        """
        fc = ast.FitsChan(ast.StringStream())
        fc.setFitsI("ANINT", 200)
        fc.setFitsS("ASTRING", "string value")

        self.assertEqual(fc.nCard, 2)

        # test getFitsX for missing cards with default values
        fv = fc.getFitsCF("BOGUS", complex(9, -5))
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, complex(9, -5))
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsCN("BOGUS", "not_there")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "not_there")
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsF("BOGUS", 55.5)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 55.5)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsI("BOGUS", 55)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 55)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsL("BOGUS", True)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, True)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsS("BOGUS", "missing")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "missing")
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        # test getFitsX for missing cards without default values
        fv = fc.getFitsCF("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, complex())
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsCN("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsF("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 0)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsI("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, 0)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsL("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, False)
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.getFitsS("BOGUS")
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")
        self.assertEqual(fc.getCardType(), ast.CardType.NOTYPE)
        self.assertEqual(fc.getCard(), fc.nCard + 1)

        fv = fc.findFits("BOGUS", inc=False)
        self.assertFalse(fv.found)
        self.assertEqual(fv.value, "")

    def test_FitsChanEmptyFits(self):
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        self.assertEqual(fc.nCard, len(self.cards))
        fc.emptyFits()
        self.assertEqual(fc.nCard, 0)

    def test_FitsChanPutCardsPutFits(self):
        ss = ast.StringStream()
        fc = ast.FitsChan(ss)
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
            fc.putFits(card, overwrite=False)
        self.assertEqual(fc.nCard, 10)
        self.assertEqual(fc.getCard(), 9)
        predCardNames = [c.split()[0] for c in self.cards[0:8]] + ["CRVAL1", "CRVAL2"]
        self.assertEqual(fc.getAllCardNames(), predCardNames)

    def test_FitsChanFindFits(self):
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        expectedNCards = fc.nCard

        # append a card with no value
        fc.setCard(fc.nCard + 1)
        fc.setFitsU("UNDEFVAL")
        expectedNCards += 1
        self.assertEqual(fc.nCard, expectedNCards)

        fc.setCard(9)  # index of CRVAL1
        self.assertEqual(fc.getCardName(), "CRVAL1")
        fv = fc.findFits("%f", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                    0"))

        # delete CRVAL1 card
        fc.delFits()
        expectedNCards -= 1
        self.assertEqual(fc.nCard, expectedNCards)
        self.assertEqual(fc.getCard(), 9)
        fv = fc.findFits("%f", inc=False)
        self.assertEqual(fc.getCard(), 9)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                    0"))

        # insert CRVAL1 card before current card; verify that the index
        # is incremented to point to the next card
        fc.putFits("CRVAL1  = 99", overwrite=False)
        expectedNCards += 1
        self.assertEqual(fc.nCard, expectedNCards)
        self.assertEqual(fc.getCard(), 10)
        fv = fc.findFits("%f", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                    0"))

        fc.setCard(9)  # index of CRVAL1
        fv = fc.findFits("%f", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                   99"))

        # overwrite CRVAL1 card
        fc.setCard(9)
        fc.putFits("CRVAL1  = 0", overwrite=True)
        self.assertEqual(fc.nCard, expectedNCards)
        fc.setCard(9)
        fv = fc.findFits("%f", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                    0"))

        # test that findFits does not wrap around
        fv = fc.findFits("CTYPE2", inc=False)
        self.assertFalse(fv.found)
        fc.clearCard()
        fv = fc.findFits("CTYPE2", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE2  = 'DEC-TAN '"))
        self.assertEqual(fc.getCard(), 4)

        # test that we can find a card with undefined value
        fc.clearCard()
        fv = fc.findFits("UNDEFVAL", inc=False)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value,
                         "UNDEFVAL=                                                                       ")

    def test_FitsChanReadWrite(self):
        ss = ast.StringStream("".join(self.cards))
        fc1 = ast.FitsChan(ss)
        obj1 = fc1.read()
        self.assertEqual(obj1.className, "FrameSet")

        ss2 = ast.StringStream()
        fc2 = ast.FitsChan(ss2, "Encoding=FITS-WCS")
        n = fc2.write(obj1)
        self.assertEqual(n, 1)
        self.assertEqual(fc2.nCard, 10)
        fc2.clearCard()

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("WCSAXES =                    2 / Number of WCS axes"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRPIX1  =                100.0 / Reference pixel on axis 1"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRPIX2  =                100.0 / Reference pixel on axis 2"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL1  =                  0.0 / Value at ref. pixel on axis 1"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CRVAL2  =                  0.0 / Value at ref. pixel on axis 2"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE1  = 'RA---TAN'           / Type of co-ordinate on axis 1"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CTYPE2  = 'DEC--TAN'           / Type of co-ordinate on axis 2"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CDELT1  =                0.001 / Pixel size on axis 1"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("CDELT2  =                0.001 / Pixel size on axis 2"))

        fv = fc2.findFits("%f", inc=True)
        self.assertTrue(fv.found)
        self.assertEqual(fv.value, pad("RADESYS = 'ICRS    '           / Reference frame for RA/DEC values"))

        self.assertEqual(ss2.getSinkData(), "")
        self.assertEqual(fc2.nCard, 10)
        fc2.writeFits()
        self.assertEqual(fc2.nCard, 0)
        a = ss2.getSinkData()

        ss3 = ast.StringStream(ss2.getSinkData())
        fc3 = ast.FitsChan(ss3, "Encoding=FITS-WCS")
        fc3.readFits()
        obj3 = fc3.read()

        ss4 = ast.StringStream()
        fc4 = ast.FitsChan(ss4, "Encoding=FITS-WCS")
        n = fc4.write(obj3)
        self.assertEqual(n, 1)
        del fc4
        b = ss4.getSinkData()
        self.assertEqual(a, b)

    def test_FitsChanTestFits(self):
        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")

        # add a card for each type
        fc.setFitsF("AFLOAT", 1.5)
        fc.setFitsS("ASTRING", "a string")
        fc.setFitsU("UNDEFVAL")

        self.assertEqual(fc.testFits("AFLOAT"), ast.PRESENT)
        self.assertEqual(fc.testFits("ASTRING"), ast.PRESENT)
        self.assertEqual(fc.testFits("UNDEFVAL"), ast.NOVALUE)
        self.assertEqual(fc.testFits("BADNAME"), ast.ABSENT)

        fc.setCard(1)
        self.assertEqual(fc.getCardName(), "AFLOAT")
        self.assertEqual(fc.testFits(), ast.PRESENT)
        fc.setCard(3)
        self.assertEqual(fc.getCardName(), "UNDEFVAL")
        self.assertEqual(fc.testFits(), ast.NOVALUE)

    def test_FitsChanInsertShift(self):
        """Check that a simple WCS can still be written as FITS-WCS
        after inserting a shift at the beginning of GRID to IWC

        This tests LSST ticket DM-12524
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss, "Encoding=FITS-WCS, IWC=1")
        frameSet = fc.read()
        self.assertIsInstance(frameSet, ast.FrameSet)
        self.assertAlmostEqual(fc.fitsTol, 0.1)

        shift = 30
        shiftMap = ast.ShiftMap([shift, shift])
        shiftedFrameSet = self.insertPixelMapping(shiftMap, frameSet)

        fc2 = writeFitsWcs(frameSet)
        self.assertGreater(fc2.nCard, 9)
        for i in (1, 2):
            fv = fc2.getFitsF("CRPIX%d" % (i,))
            self.assertAlmostEqual(fv.value, 100)

        fc3 = writeFitsWcs(shiftedFrameSet)
        self.assertGreaterEqual(fc3.nCard, fc2.nCard)
        for i in (1, 2):
            fv = fc3.getFitsF("CRPIX%d" % (i,))
            self.assertAlmostEqual(fv.value, 100 - shift)
        for name in fc2.getAllCardNames():
            self.assertEqual(fc3.testFits(name), ast.PRESENT)

    def test_FitsChanFitsTol(self):
        """Test that increasing FitsTol allows writing a WCS with distortion as FITS-WCS
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss, "Encoding=FITS-WCS, IWC=1")
        frameSet = fc.read()

        distortion = ast.PcdMap(0.001, [0.0, 0.0])
        distortedFrameSet = self.insertPixelMapping(distortion, frameSet)

        # Writing as FTIS-WCS should fail with the default FitsTol
        fc = writeFitsWcs(distortedFrameSet)
        self.assertEqual(fc.nCard, 0)

        # Writing as FITS-WCS should succeed with adequate FitsTol
        fc2 = writeFitsWcs(distortedFrameSet, "FitsTol=1000")
        self.assertGreater(fc2.nCard, 9)
        for i in (1, 2):
            fv = fc2.getFitsF("CRVAL%d" % (i,))
            self.assertAlmostEqual(fv.value, 0)
            fv2 = fc2.getFitsS("CTYPE1")
            self.assertEqual(fv2.value, "RA---TAN")
            fv3 = fc2.getFitsS("CTYPE2")
            self.assertEqual(fv3.value, "DEC--TAN")

    def test_FitsChanDM13686(self):
        """Test that a particular FrameSet will not segfault when
        we attempt to write it to a FitsChan as FITS-WCS
        """
        def readObjectFromShow(path):
            """Read an ast object saved as Object.show()"""
            with open(path, "r") as f:
                objectText = f.read()
            stream = ast.StringStream(objectText)
            chan = ast.Channel(stream)
            return chan.read()

        path = os.path.join(self.dataDir, "frameSetDM13686.txt")
        frameSet = readObjectFromShow(path)
        strStream = ast.StringStream()
        fitsChan = ast.FitsChan(strStream, "Encoding=FITS-WCS")
        # This FrameSet can be represtented as FITS-WCS, so 1 object is written
        self.assertEqual(fitsChan.write(frameSet), 1)

    def test_FitsChanTAB(self):
        """Test that FITS -TAB WCS can be created.
        """

        wavelength = np.array([0., 0.5, 1.5, 3., 5.])

        # Create a FrameSet using a LutMap with non-linear coordinates
        pixelFrame = ast.Frame(1, "Domain=PIXELS")
        wavelengthFrame = ast.SpecFrame("System=wave, unit=nm")
        lutMap = ast.LutMap(wavelength, 1, 1)
        frameSet = ast.FrameDict(pixelFrame)
        frameSet.addFrame("PIXELS", lutMap, wavelengthFrame)

        # Now serialize it using -TAB WCS
        fc = writeFitsWcs(frameSet, "TabOk=1")

        fv = fc.getFitsS("CTYPE1")
        self.assertEqual(fv.value, "WAVE-TAB")

        # PS1_0 is the table extension name
        fv = fc.getFitsS("PS1_0")
        waveext = fv.value
        self.assertEqual(waveext, "WCS-TAB")

        # PS1_1 is the column name for the wavelength
        fv = fc.getFitsS("PS1_1")
        wavecol = fv.value
        self.assertEqual(wavecol, "COORDS1")

        # Get the WCS table from the FitsChan
        km = fc.getTables()
        table = km.getA(waveext, 0)
        fc_bintab = table.getTableHeader()

        fv = fc_bintab.getFitsS("TDIM1")
        self.assertEqual(fv.value, "(1,5)")

        self.assertEqual(table.nRow, 1)
        self.assertEqual(table.nColumn, 1)

        # 1-based column numbering to match FITS
        cname = table.columnName(1)
        self.assertEqual(cname, "COORDS1")
        self.assertEqual(table.columnType(cname), ast.DataType.DoubleType)
        self.assertEqual(table.columnSize(cname), 40)
        self.assertEqual(table.columnNdim(cname), 2)
        self.assertEqual(table.columnUnit(cname), "nm")
        self.assertEqual(table.columnLength(cname), 5)
        self.assertEqual(table.columnShape(cname), [1, 5])
        coldata = table.getColumnData1D(cname)
        self.assertEqual(list(coldata), list(wavelength))

        # This will be shaped correctly as a numpy array with third dimension
        # the row count.
        coldata = table.getColumnData(cname)
        self.assertEqual(coldata.ndim, 3)
        self.assertEqual(coldata.shape, (1, 5, 1))

    def test_python(self):
        """Test Python Mapping/Sequence interface to FitsChan.
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        self.assertEqual(len(fc), 18)
        cards = "".join(c for c in fc)

        self.assertEqual(cards, "".join(self.cards))
        self.assertIn("CTYPE2", fc)
        self.assertIn(10, fc)
        self.assertNotIn(-1, fc)
        self.assertNotIn(20, fc)
        self.assertNotIn("CTYPE3", fc)

        self.assertEqual(fc["CTYPE1"], "RA--TAN")
        self.assertEqual(fc["NAXIS2"], 200)
        self.assertEqual(fc["CDELT2"], 0.001)
        self.assertFalse(fc["BOOL"])
        self.assertEqual(fc[4].rstrip(), "CRPIX1  =                  100")
        with self.assertRaises(KeyError):
            fc["NOTIN"]
        with self.assertRaises(IndexError):
            fc[100]

        # Update values
        fc["BOOL"] = True  # This will remove second card
        self.assertEqual(fc["BOOL"], True)
        fc["CRVAL2"] = None
        self.assertIsNone(fc["CRVAL2"])
        fc["NEWSTR"] = "Test"
        self.assertEqual(fc["NEWSTR"], "Test")
        fc["NEWINT"] = 1024
        self.assertEqual(fc["NEWINT"], 1024)
        fc["NEWFLT"] = 3.5
        self.assertEqual(fc["NEWFLT"], 3.5)
        fc["UNDEF"] = "not undef"
        self.assertEqual(fc["UNDEF"], "not undef")

        fc[""] = "A new BLANK comment"
        fc[0] = "COMMENT Introduction comment"
        self.assertEqual(fc[0].rstrip(), "COMMENT Introduction comment")

        # This will fail since a string is required
        with self.assertRaises(TypeError):
            fc[0] = 52

        # Delete the 3rd card
        del fc[2]
        self.assertEqual(fc[2].rstrip(), "CTYPE2  = 'DEC-TAN '")
        self.assertEqual(len(fc), 20)

        # Delete all the HISTORY cards
        del fc["HISTORY"]
        self.assertEqual(len(fc), 17)

        # Change a card to blank
        fc[3] = None
        self.assertEqual(fc[3].strip(), "")
        fc[3] = "COMMENT 1"
        self.assertEqual(fc[3].strip(), "COMMENT 1")
        fc[3] = ""
        self.assertEqual(fc[3].strip(), "")

        # Use negative index
        self.assertEqual(fc[-1], fc[fc.nCard-1])
        fc[-2] = "COMMENT new comment"
        self.assertEqual(fc[-2], fc[fc.nCard-2])
        self.assertEqual(fc[-2].rstrip(), "COMMENT new comment")

        # Append a comment to the end
        nCards = len(fc)
        fc[fc.nCard] = "COMMENT X"
        self.assertEqual(len(fc), nCards + 1)
        self.assertEqual(fc[-1].rstrip(), "COMMENT X")

        # Try to access and append using a high index
        with self.assertRaises(IndexError):
            fc[fc.nCard + 1]
        with self.assertRaises(IndexError):
            fc[fc.nCard + 1] = ""
        with self.assertRaises(IndexError):
            fc[fc.nCard]

        # Or the wrong type of key
        with self.assertRaises(KeyError):
            fc[3.14] = 52

        # Delete final card
        nCards = len(fc)
        del fc[-1]
        self.assertEqual(fc[-1].rstrip(), "NEWFLT  =                  3.5")
        self.assertEqual(len(fc), nCards - 1)

        with self.assertRaises(IndexError):
            del fc[-fc.nCard - 1]

        with self.assertRaises(IndexError):
            del fc[fc.nCard]

        with self.assertRaises(KeyError):
            del fc[3.14]

        with self.assertRaises(KeyError):
            del fc["NOTTHERE"]

        # Test stringification
        header = str(fc)
        self.assertIn("BOOL    =                    1", header)

        # All the items
        collected = []
        for k, v in fc.items():
            collected.append((k, v))
        self.assertEqual(len(collected), len(fc))


if __name__ == "__main__":
    unittest.main()
