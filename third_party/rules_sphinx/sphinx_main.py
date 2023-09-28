import os
import sys

"""Wraps `sphinx-build`, all arguments are passed on to `sphinx-build`.
"""

if __name__ == "__main__":
    # Seemingly unused imports; but is to overcome error with namespaced Python packages (in sphinxcontrib.*) that would
    # otherwise not be importable.
    import sphinxcontrib  # noqa: F401

    # 'Sniff' the sources input directory as a way to know what it would be a configuration-time in conf.py.
    # Needed for pointing to custom CSS/JS.
    os.environ["BAZEL_SPHINX_INPUT_DIR"] = os.path.abspath(sys.argv[-2])

    from sphinx.cmd.build import main

    sys.exit(main(sys.argv[1:]))
