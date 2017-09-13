from __future__ import absolute_import, division, print_function
import os.path
import unittest

import astshim as ast
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
            "COMMENT  one of two comments",
            "COMMENT  another of two comments",
            "HISTORY  one of two history fields",
            "HISTORY  second of three history fields",
            "HISTORY  third of three history fields",
        )
        self.cards = [pad(card) for card in shortCards]

    def test_FitsChanPreloaded(self):
        """Test a FitsChan that starts out loaded with data
        """
        ss = ast.StringStream("".join(self.cards))
        fc = ast.FitsChan(ss)
        self.assertEqual(fc.nCard, len(self.cards))
        # there are 2 COMMENT and 3 HISTORY cards, so 3 fewer unique keys
        self.assertEqual(fc.nKey, len(self.cards) - 3)
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
        path = os.path.join(DataDir, "test_fitsChanFileStream.fits")
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
        fc.setFitsCF("NEWNAME", complex(99.9, 99.8), "New comment", overwrite = True)
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


if __name__ == "__main__":
    unittest.main()
