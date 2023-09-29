# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import csv
import os
import re
import sys
from datetime import datetime
from types import ModuleType
from typing import Any, Dict, List, Optional, Set

from snowflake.ml.version import VERSION

# -- Project information -----------------------------------------------------

project = "Snowpark ML API Reference (Python)"
copyright = f"{datetime.now().year} Snowflake Inc."
author = "Snowflake Inc."
release = VERSION

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.coverage",
    # "sphinx.ext.linkcode"
]

html_theme_options = {"rightsidebar": "true", "relbarbgcolor": "black"}

# -- Options for autodoc --------------------------------------------------
autodoc_default_options = {
    "autosummary-generate": True,
    "member-order": "alphabetical",  # 'alphabetical', by member type ('groupwise') or source order (value 'bysource')
    "undoc-members": True,  # If set, autodoc will also generate document for the members not having docstrings
    "show-inheritance": True,
}

autodoc_mock_imports: List[str] = []
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*/snowflake/snowpark/**", "*/snowflake/connector/**"]

autosummary_generate = True
autosummary_generate_overwrite = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme_path = [
    os.environ["BAZEL_SPHINX_INPUT_DIR"] + "/_themes",
]

html_theme = "empty"

html_theme_options = {
    # 'analytics_id': 'UA-XXXXXXX-1',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = [os.environ["BAZEL_SPHINX_INPUT_DIR"] + "/_static"]
templates_path = [os.environ["BAZEL_SPHINX_INPUT_DIR"] + "/_templates"]


html_show_sourcelink = False  # Hide "view page source" link

# Disable footer message "Built with Sphinx using a theme provided by Read the Docs."
html_show_sphinx = False

suppress_warnings = ["ref"]


def setup(app: Any) -> None:
    app.connect(
        "autodoc-skip-member",
        SkipMember(os.path.join(os.path.dirname(os.path.realpath(__file__)), "unsupported_functions_by_class.csv")),
    )
    app.connect("autodoc-process-docstring", fix_markdown_links)

    # fix up numpy typing so that we can import certain modules
    import numpy
    import numpy._typing

    if not hasattr(numpy._typing, "_DType"):
        numpy._typing._DType = numpy.dtype


# Construct URL to corresponding section in the GitHub repo
# Not currently used
def linkcode_resolve(domain: str, info: Dict[str, Any]) -> Optional[str]:
    import inspect

    # import pkg_resources
    # import snowflake.ml
    if domain != "py":
        return None

    mod_name = info["module"]
    full_name = info["fullname"]

    obj = sys.modules.get(mod_name)
    if obj is None:
        return None

    for part in full_name.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        if isinstance(obj, property):
            assert obj.fget is not None
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        else:
            fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        return None

    try:
        if isinstance(obj, property):
            assert obj.fget is not None
            source, lineno = inspect.getsourcelines(obj.fget)
        else:
            source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    except TypeError:
        linespec = ""
    if fn:
        return (
            f"https://github.com/snowflakedb/snowflake-ml-python/blob/"
            f"{release}/{os.path.relpath(fn, start=os.pardir)}{linespec}"
        )
    return None


# class that skips documenting unsupported methods of certain classes
# the list of unsupported methods is read from a CSV file
# also eliminates documentation for dependent classes such as Snowpark Python classes
class SkipMember:
    class_name: str = ""
    unsupported_methods: Dict[str, Set[str]] = {}

    def __init__(self, csv_filename: str) -> None:
        self.unsupported_methods = {}
        for row in csv.reader(open(csv_filename)):
            class_name = row[0].strip()
            unsupported_methods = {col.strip() for col in row[1:]}
            self.unsupported_methods[class_name] = unsupported_methods

    # sphinx expects a function for this, so make instance callable
    def __call__(self, app: Any, what: str, name: str, obj: ModuleType, skip: bool, options: Dict[str, Any]) -> bool:
        if name.startswith("_"):
            return True
        if what == "method":
            class_name, method_name = obj.__qualname__.split(".")
            return not obj.__module__.startswith("snowflake.ml") or name in self.unsupported_methods.get(class_name, ())
        return False


# some links in the docstrings are written as Markdown, sometimes broken by whitespace.
# convert these to RST links by brute force (meaning regex)
def fix_markdown_links(app: Any, what: str, name: str, obj: object, options: Dict[str, Any], lines: List[str]) -> None:
    docstring = "\n".join(lines)
    # remove whitespace between parts of markdown-style links
    docstring = re.sub(r"\]\s+\(http", "](http", docstring, flags=re.MULTILINE)
    # convert markdown-style links to RST-style links
    docstring = re.sub(r"\[(.*?)\]\((http.*?)\)", r"`\1 <\2>`_", docstring)
    # while we're here, also re-point lightgbm links to the latest stable doc like scikit-learn and xgboost
    docstring = docstring.replace(
        "<https://lightgbm.readthedocs.io/en/v3.3.2/", "<https://lightgbm.readthedocs.io/en/stable/"
    )
    # pass them back to Sphinx
    lines[:] = docstring.splitlines()
