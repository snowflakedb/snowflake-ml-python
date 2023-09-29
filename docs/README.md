# Snowpark ML Docs README

## Build the docs

To start, please follow [CONTRIBUTING.md](/CONTRIBUTING.md) to setup your `bazel`.

To build the docs, use

```bash
bazel build docs --config=docs
```

Upon a successful build, view the built HTML doc by opening `$(bazel info bazel-bin)/docs/html/index.html` in a Web browser.
For example,

```bash
open $(bazel info bazel-bin)/docs/html/index.html
```

## Important files and directories

The following files are in the `docs` directory.

- `BUILD.bazel`: Defines bazel rules to build the docs.
- `source` directory: Source files for docs including templates, themes, hand-written rst files.
- `sphinxconf` directory: The configuration directory of the docs building.
  - `conf.py`: The configuration parameters for Sphinx and the automatic documentation modules.
  - `unsupported_functions_by_class.csv`: CSV file indicating methods that should not be documented in specific classes.

The following files are in the `docs/source` directory:

- `index.rst`: ReStructuredText (RST) file that will be built as the index page.
  It mainly as a landing point and indicates the subp-ackages to include in the API reference.
  Currently these include the Modeling and FileSet/FileSystem APIs.
- `fileset.rst`, `modeling.rst`, `registry.rst`: RST files that direct Sphinx to include the specific classes in each submodule.
