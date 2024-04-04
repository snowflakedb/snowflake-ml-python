"""Implement a generator for human readable ID (HRID).

The original idea for this comes from Asana where it is documented on their
blog:
    http://blog.asana.com/2011/09/6-sad-squid-snuggle-softly/

There are other partial implementations of this and can be found here:
    Node.js: https://github.com/linus/greg
    Java: https://github.com/PerWiklander/IdentifierSentence

In this module you will find:

    HRID16: An implementation of HRIDBase for 16 bit integers.

The list used here is coming from:
    https://git.coolaj86.com/coolaj86/human-readable-ids.js
"""

import random

import importlib_resources

from snowflake.ml._internal import human_readable_id
from snowflake.ml._internal.human_readable_id import hrid_generator_base


class HRID16(hrid_generator_base.HRIDBase):
    """An implementation of HRIDBase for 16 bit integers."""

    def __id_generator__(self) -> int:
        return int(random.getrandbits(16))

    __hrid_structure__ = ("adjective", "animal", "number")
    __hrid_words__ = dict(
        number=tuple(str(x) for x in range(1, 5)),
        adjective=tuple(
            importlib_resources.files(human_readable_id).joinpath("adjectives.txt").read_text("utf-8").split()
        ),
        animal=tuple(importlib_resources.files(human_readable_id).joinpath("animals.txt").read_text("utf-8").split()),
    )
