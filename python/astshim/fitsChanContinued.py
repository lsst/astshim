from .fitsChan import FitsChan, CardType
from .fitsChan import *  # noqa: F403 F401


def length(self):
    return self.nCard


FitsChan.__len__ = length


def iter(self):
    """The FitsChan itself is the iterator."""
    self.clearCard()
    return self


FitsChan.__iter__ = iter


def next(self):
    """Return the entire header card until we run out of cards"""
    card = self.findFits("%f", True)
    if not card.found:
        raise StopIteration
    return card.value


FitsChan.__next__ = next


def to_string(self):
    """A FitsChan string representation is a FITS header with newlines
    after each header card."""
    return "\n".join(c for c in self)


FitsChan.__str__ = to_string


def contains(self, name):
    """Returns True if either the supplied name is present in the FitsChan
    or the supplied integer is acceptable to the FitsChan."""
    if isinstance(name, int):
        # index will be zero-based
        if name >= 0 and name < self.nCard:
            return True
    elif isinstance(name, str):
        currentCard = self.getCard()
        try:
            self.clearCard()
            result = self.findFits(name, False)
        finally:
            self.setCard(currentCard)
        if result.found:
            return True

    return False


FitsChan.__contains__ = contains


def getitem(self, name):
    """Return the value if a keyword is specified, or the entire card if
    an integer index is specified.

    Returns a single value when integer index is used, returns a tuple
    if a card name is used (since a FITS header can contain multiple
    cards with the same name).
    """

    # Save current card position
    currentCard = self.getCard()

    if isinstance(name, int):
        # FITS is 1-based and Python is zero-based so add 1
        self.setCard(name + 1)
        try:
            result = self.findFits("%f", False)
        finally:
            self.setCard(currentCard)
        if not result.found:
            raise IndexError(f"No FITS card at index {name}")
        return result.value

    elif isinstance(name, str):

        try:
            # Rewind FitsChan so we search all cards
            self.clearCard()

            # Method look up table for obtaining values
            typeLut = {CardType.INT: self.getFitsI,
                       CardType.FLOAT: self.getFitsF,
                       CardType.STRING: self.getFitsS,
                       CardType.COMPLEXF: self.getFitsCF,
                       CardType.LOGICAL: self.getFitsL
                       }

            # We can have multiple matches
            values = []

            # Loop over each item that matches
            while True:
                result = self.findFits(name, False)
                if not result.found:
                    break

                # Get the data type for this matching card
                ctype = self.getCardType()

                # Go back one card so we can ask for the value in the correct
                # data type (getFitsX starts from the next card)
                thiscard = self.getCard()
                self.setCard(thiscard - 1)

                if ctype == CardType.UNDEF:
                    values.append(None)
                elif ctype in typeLut:
                    found = typeLut[ctype](name)
                    if found.found:
                        values.append(found.value)
                    else:
                        raise RuntimeError(f"Unexpectedly failed to find card '{name}'")
                else:
                    raise RuntimeError(f"Type, {ctype} of FITS card '{name}' not supported")

                # Increment the card number to continue search
                self.setCard(thiscard + 1)
        finally:
            # Reinstate the original card position
            self.setCard(currentCard)

        # We may have multiple values. For consistency with pyast we return
        # a single item for a single match, else a tuple
        result = None
        if not values:
            raise KeyError(f"{name}")

        if len(values) == 1:
            result = values[0]
        else:
            result = tuple(values)

        return result

    raise ValueError(f"Supplied key, '{name}' of unsupported type")


FitsChan.__getitem__ = getitem


def setitem(self, name, value):
    """name can be integer index or keyword.

    If an integer, the value is deemed to be a full card to replace the
    existing one. If the value is None or empty string the card is deleted.

    This can affect the position of the current card since cards can be
    inserted."""

    if isinstance(name, int):
        # Correct to 1-based
        self.setCard(name + 1)

        # Overwrite the current card or delete it
        if value:
            self.putFits(value, True)
        else:
            self.delFits()
        return

    # A blank name results in a comment card being inserted at the
    # current position
    if not name:
        self.setFitsCM(value, False)
        return

    # Get current card position and rewind
    currentCard = self.getCard()
    try:
        self.clearCard()

        # Look for a card with the specified name
        # We do not care about the result, if nothing is found we will be at the
        # end of the header
        self.findFits(name, False)

        # pyast seems to want to delete items if the value is None but
        # astropy and PropertyList think the key should be undefined.
        if value is None:
            self.setFitsU(name, overwrite=True)
        elif isinstance(value, int):
            self.setFitsI(name, value, overwrite=True)
        elif isinstance(value, float):
            self.setFitsF(name, value, overwrite=True)
        elif isinstance(value, bool):
            self.setFitsL(name, value, overwrite=True)
        else:
            # Force to string
            self.setFitsS(name, str(value), overwrite=True)

        # Delete any later cards with matching keyword
        while True:
            found = self.findFits(name, False)
            if not found.found:
                break
            self.delFits()
    finally:
        # Try to reinstate the current card
        self.setCard(currentCard)


FitsChan.__setitem__ = setitem
