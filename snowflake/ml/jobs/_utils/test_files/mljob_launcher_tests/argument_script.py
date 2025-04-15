# Script that uses command-line arguments
import sys


def process_args(pos_arg=None, named_arg=None):
    return {"pos_args": sys.argv[1:], "provided_pos_arg": pos_arg, "provided_named_arg": named_arg}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        __return__ = {"args": sys.argv[1:]}
    else:
        __return__ = {"args": []}
