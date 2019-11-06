from .fitsChan import FitsChan, CardType
from .fitsChan import *  # noqa: F403 F401


def _calc_card_pos(self, index):
    """Convert a python index into a FitsChan position.

    Parameters
    ----------
    self : `FitsChan`
        The FitsChan to index.
    index : `int`
        0-based index into header. If negative, counts from end.

    Raises
    ------
    IndexError
        Raised if the index exceeds the size of the FitsChan. If the index
        equals the size of the FitsChan (noting that in 0-based indexing the
        final card is one less than the size) then this refers to the end of
        the header.
    """
    # Calculate 0-based index
    nCards = len(self)
    if index < 0:
        index = nCards + index
    elif abs(index) > nCards:
        # We allow index of one higher to indicate append
        raise IndexError(f"Index {index} exceeds size of FitsChan ({nCards})")

    # Convert to 1-based index
    return index + 1


def _get_current_card_value(self):
    """Retrieve the value of the current card.

    Returns
    -------
    name : `str`
        The name of the current card.
    value : `object`
        The value in the correct Python type.
    """
    # Method look up table for obtaining values
    typeLut = {CardType.INT: self.getFitsI,
               CardType.FLOAT: self.getFitsF,
               CardType.STRING: self.getFitsS,
               CardType.COMPLEXF: self.getFitsCF,
               CardType.LOGICAL: self.getFitsL
               }

    # Get the data type for this matching card
    ctype = self.getCardType()

    # Get the name of the card
    name = self.getCardName()

    # Go back one card so we can ask for the value in the correct
    # data type (getFitsX starts from the next card)
    # thiscard = self.getCard()
    # self.setCard(thiscard - 1)

    if ctype == CardType.UNDEF:
        value = None
    elif ctype in typeLut:
        found = typeLut[ctype]("")  # "" indicates current card
        if found.found:
            value = found.value
        else:
            raise RuntimeError(f"Unexpectedly failed to find card '{name}'")
    elif ctype == CardType.COMMENT:
        value = self.getCardComm()
    else:
        raise RuntimeError(f"Type, {ctype} of FITS card '{name}' not supported")

    return name, value


def length(self):
    return self.nCard


FitsChan.__len__ = length


def iter(self):
    """The FitsChan itself is the iterator.

    The position of the iterator is handled internally in the FitsChan and
    is moved to the start of the FitsChan by this call.
    Whilst iterating do not change the internal card position.
    """
    self.clearCard()
    return self


FitsChan.__iter__ = iter


def next(self):
    """Return each 80-character header card until we run out of cards.
    """
    card = self.findFits("%f", True)
    if not card.found:
        raise StopIteration
    return card.value


FitsChan.__next__ = next


def to_string(self):
    """A FitsChan string representation is a FITS header with newlines
    after each header card.
    """
    return "\n".join(c for c in self)


FitsChan.__str__ = to_string


def contains(self, name):
    """Returns True if either the supplied name is present in the FitsChan
    or the supplied integer is acceptable to the FitsChan.
    """
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
        # Calculate position in FitsChan (0-based to 1-based)
        newpos = _calc_card_pos(self, name)
        self.setCard(newpos)
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

            # We are only interested in the first matching card
            result = self.findFits(name, False)
            if not result.found:
                raise KeyError(f"{name}'")

            this_name, value = _get_current_card_value(self)
            if this_name != name:
                raise RuntimeError(f"Internal inconsistency in get: {this_name} != {name}")

        finally:
            # Reinstate the original card position
            self.setCard(currentCard)

        return value

    raise ValueError(f"Supplied key, '{name}' of unsupported type")


FitsChan.__getitem__ = getitem


def setitem(self, name, value):
    """name can be integer index or keyword.

    If an integer, the value is deemed to be a full card to replace the
    existing one. If the value is None or empty string a blank header
    card is created at the location.

    This can affect the position of the current card since cards can be
    inserted."""

    if isinstance(name, int):
        # Calculate position in FitsChan (0-based to 1-based)
        newpos = _calc_card_pos(self, name)
        self.setCard(newpos)

        if not value:
            value = " "

        # Overwrite the entire value
        self.putFits(value, True)
        return

    # A blank name results in a comment card being inserted at the
    # current position
    if not name:
        self.setFitsCM(value, False)
        return

    if not isinstance(name, str):
        raise KeyError(f"Supplied key, '{name}' of unsupported type")

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


def delitem(self, name):
    """Delete an item from the FitsChan either by index (0-based) or by name.

    If a name is given all instances of the name will be deleted.
    The current card position is not retained.
    """
    if isinstance(name, int):
        # Correct to 1-based
        newpos = _calc_card_pos(self, name)
        if newpos <= 0 or newpos > self.nCard:
            # AST will ignore this but we raise if the index is out of range
            raise IndexError(f"No FITS card at index {name}")
        self.setCard(newpos)
        self.delFits()
        return

    if not isinstance(name, str):
        raise KeyError(f"Supplied key, '{name}' of unsupported type")

    self.clearCard()
    # Delete any cards with matching keyword
    deleted = False
    while True:
        found = self.findFits(name, False)
        if not found.found:
            break
        self.delFits()
        deleted = True

    if not deleted:
        raise KeyError(f"No FITS card named {name}")


FitsChan.__delitem__ = delitem


def items(self):
    """Iterate over each card, returning the keyword and value in a tuple.

    The position of the iterator is internal to the FitsChan.  Do not
    change the card position when iterating.
    """
    self.clearCard()
    nCards = self.nCard
    thisCard = self.getCard()

    while thisCard <= nCards:
        name, value = _get_current_card_value(self)
        yield name, value
        self.setCard(thisCard + 1)
        thisCard = self.getCard()


FitsChan.items = items
