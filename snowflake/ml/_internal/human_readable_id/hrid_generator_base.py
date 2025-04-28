"""Implement a generator for human readable ID (HRID).

The original idea for this comes from Asana where it is documented on their
blog:
    http://blog.asana.com/2011/09/6-sad-squid-snuggle-softly/

There are other partial implementations of this and can be found here:
    Node.js: https://github.com/linus/greg
    Java: https://github.com/PerWiklander/IdentifierSentence

In this module you will find:

    HRIDBase: The base class for all human readable id.
"""

import math
from abc import ABC, abstractmethod


class HRIDBase(ABC):
    """The base class for all all human readable id.

    This provides all of the necessary helper functionality to turn IDs into
    HRIDs and HRIDs into IDs. ID typically is a random int, while HRID is a corresponding short string.
    """

    @abstractmethod
    def __id_generator__(self) -> int:
        """The generator to use to generate new IDs. The implementer needs to provide this."""

    __hrid_structure__: tuple[str, ...]
    """The HRID structure to be generated. The implementer needs to provide this."""

    __hrid_words__: dict[str, tuple[str, ...]]
    """The mapping between the HRID parts and the words to use. The implementer needs to provide this."""

    __separator__ = "_"

    def __init__(self) -> None:
        self._part_n_words = dict()
        self._part_bits = dict()
        for part in self.__hrid_structure__:
            n_words = len(self.__hrid_words__[part])
            self._part_n_words[part] = n_words
            if not (n_words > 0 and ((n_words & (n_words - 1)) == 0)):
                raise ValueError(f"{part} part has {n_words} words, which is not a power of 2")
            self._part_bits[part] = int(math.log(self._part_n_words[part], 2))
        self.__total_bits__ = sum(v for v in self._part_bits.values())

    def hrid_to_id(self, hrid: str) -> int:
        """Take the HRID and convert it the ID.

        Args:
            hrid: The HRID to convert into an ID

        Returns:
            The ID represented by the HRID
        """
        idxs = self._hrid_to_idxs(hrid)
        id = 0
        for i in range(len(idxs)):
            part = self.__hrid_structure__[i]
            id = (id << self._part_bits[part]) + idxs[i]
        return id

    def id_to_hrid(self, id: int) -> str:
        """Take the ID and convert it a HRID.

        Args:
            id: The ID to convert into a HRID

        Returns:
            The HRID represented by the ID
        """
        idxs = self._id_to_idxs(id)
        hrid = []
        for i in range(len(self.__hrid_structure__)):
            part = self.__hrid_structure__[i]
            values = self.__hrid_words__[part]
            hrid.append(str(values[idxs[i]]))
        return self.__separator__.join(hrid)

    def generate(self) -> tuple[int, str]:
        """Generate an ID and the corresponding HRID.

        Returns:
            A tuple containing the id and the HRID
        """
        id = self.__id_generator__()
        hrid = self.id_to_hrid(id)
        return (id, hrid)

    def _id_to_idxs(self, id: int) -> list[int]:
        """Take the ID and convert it to indices into the HRID words.

        Args:
            id: The ID to convert into indices

        Returns:
            A list of indices into the HRID words
        """
        shift = self.__total_bits__
        idxs = []
        for part in self.__hrid_structure__:
            shift -= self._part_bits[part]
            mask = (self._part_n_words[part] - 1) << shift
            idxs.append((id & mask) >> shift)
        return idxs

    def _hrid_to_idxs(self, hrid: str) -> list[int]:
        """Take the HRID and convert it to indices into the HRID words.

        Args:
            hrid: The HRID to convert into indices

        Raises:
            ValueError: Raised when the input does not meet the structure.

        Returns:
            A list of indices into the HRID words
        """
        split_hrid = hrid.split(self.__separator__)
        if len(split_hrid) != len(self.__hrid_structure__):
            raise ValueError(
                ("The hrid must have {} parts and be of the form {}").format(
                    len(self.__hrid_structure__), self.__hrid_structure__
                )
            )
        idxs = []
        for i in range(len(self.__hrid_structure__)):
            part = self.__hrid_structure__[i]
            idxs.append(self.__hrid_words__[part].index(split_hrid[i]))
        return idxs
