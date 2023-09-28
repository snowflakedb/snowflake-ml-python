"""py_{binary|library|test} rules for snowml repository.

Overriding default implementation of py_{binary|library|test} to add additional features:
1. Codebase is split between 'src' & 'tests' top-level directories. We did not want python import with
   'src.snowflake.ml...'. We just want 'snowflake.ml....'. Hence I added to add additional imports statements
   to `py_library`.
2. Added check so that a target in `src` cannot depend on something in `tests`.
3. Similarly, added a check that a non-experimental target cannot depend on `experimental` target.
4. A boolean attribute "compatible_with_snowpark" is available in all the wrapped rules.
   The value of this flag affects which platform (semantically a platform maps to a conda
   environment; there are two platforms -- a snowflake channel only environment and an extended
   environment) is valid for a target. The choice of platform also affect the Python toolchain (the
   actual conda environment used to build, run and test, see WORKSPACE).
   - Targets that has compatible_with_snowpark=False, or has any (transitive)
     dependency whose compatible_with_snowpark is False, can only be built with
     //bazel/platforms:extended_conda_env.
   - All other targets can be built with either //bazel/platforms:extended_conda_env or
     //bazel/platforms:snowflake_conda_env
   - None of the target can be built with the host platform (@local_config_platform//:host). However
     that platform is not the default platform (see .bazelrc).

### Setup
```python
load("//bazel:py_rules.bzl", "py_binary", "py_library", "py_test")
```
"""

load(
    "@rules_python//python:defs.bzl",
    native_py_binary = "py_binary",
    native_py_library = "py_library",
    native_py_test = "py_test",
)
load("@rules_python//python:packaging.bzl", native_py_wheel = "py_wheel")
load(":repo_paths.bzl", "check_for_experimental_dependencies", "check_for_test_name", "check_for_tests_dependencies")

def py_genrule(**attrs):
    original_cmd = attrs["cmd"]
    attrs["cmd"] = select({
        "@bazel_tools//src/conditions:windows": "CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1 " + original_cmd,
        "//conditions:default": original_cmd,
    })
    native.genrule(**attrs)

_COMPATIBLE_WITH_SNOWPARK_TAG = "wheel_compatible_with_snowpark"

def _add_target_compatibility_labels(compatible_with_snowpark, attrs):
    if compatible_with_snowpark:
        attrs["target_compatible_with"] = select({
            "//bazel/platforms:extended_conda_channels": [],
            "//bazel/platforms:snowflake_conda_channel": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })
    else:
        attrs["target_compatible_with"] = select({
            "//bazel/platforms:extended_conda_channels": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })

def py_binary(compatible_with_snowpark = True, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_binary](https://docs.bazel.build/versions/master/be/python.html#py_binary) documentation.

    Args:
      compatible_with_snowpark: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    attrs["env"] = select({
        "@bazel_tools//src/conditions:windows": {"CONDA_DLL_SEARCH_MODIFICATION_ENABLE": "1"},
        "//conditions:default": {},
    })
    native_py_binary(**attrs)

def py_library(compatible_with_snowpark = True, **attrs):
    """Modified version of core py_library to add additional imports and check for experimental dependencies.

    See the Bazel core [py_library](https://docs.bazel.build/versions/master/be/python.html#py_library) documentation.

    Additional import is necessary to expose a library outside of top-level src directory. For example,
    by defining `//src/snowflake/ml/utils:connection_params` as follows:
    ```
    py_library(
        name = "connection_params",
        srcs = ["connection_params.py"],
        imports = ["../../.."],
        deps = [],
    )
    ```
    make the library available at the top-level, which is default `PYTHONPATH` under `bazel`. Now, `connection_params`
    is visible to anyone as `snowflake.ml.utils.connection_params`. This rule automates the generation of additional
    imports based current package's depth.

    Args:
      compatible_with_snowpark: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, attrs)

    native_py_library(**attrs)

def py_test(compatible_with_snowpark = True, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_test](https://docs.bazel.build/versions/master/be/python.html#py_test) documentation.

    Args:
      compatible_with_snowpark: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_test_name(native.package_name(), attrs):
        fail("A test target does not have a valid name!")
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    native_py_test(**attrs)

def _path_inside_wheel(input_file):
    # input_file.short_path is sometimes relative ("../${repository_root}/foobar")
    # which is not a valid path within a zip file. Fix that.
    short_path = input_file.short_path
    if short_path.startswith("..") and len(short_path) >= 3:
        # Path separator. '/' on linux.
        separator = short_path[2]

        # Consume '../' part.
        short_path = short_path[3:]

        # Find position of next '/' and consume everything up to that character.
        pos = short_path.find(separator)
        short_path = short_path[pos + 1:]
    return short_path

def _py_package_impl(ctx):
    inputs = depset(
        transitive = [dep[DefaultInfo].data_runfiles.files for dep in ctx.attr.deps] +
                     [dep[DefaultInfo].default_runfiles.files for dep in ctx.attr.deps],
    )

    # TODO: '/' is wrong on windows, but the path separator is not available in starlark.
    # Fix this once ctx.configuration has directory separator information.
    packages = [p.replace(".", "/") for p in ctx.attr.packages]
    if not packages:
        filtered_inputs = inputs
    else:
        filtered_files = []

        # TODO: flattening depset to list gives poor performance,
        for input_file in inputs.to_list():
            wheel_path = _path_inside_wheel(input_file)
            for package in packages:
                if wheel_path.startswith(package):
                    filtered_files.append(input_file)
        filtered_inputs = depset(direct = filtered_files)

    return [
        DefaultInfo(
            files = filtered_inputs,
            default_runfiles = ctx.runfiles(inputs.to_list(), collect_default = True),
        ),
        PyInfo(
            transitive_sources = depset(transitive = [
                dep[PyInfo].transitive_sources
                for dep in ctx.attr.deps
            ]),
            imports = depset(transitive = [
                dep[PyInfo].imports
                for dep in ctx.attr.deps
            ]),
        ),
    ]

py_package_lib = struct(
    implementation = _py_package_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "",
        ),
        "packages": attr.string_list(
            mandatory = False,
            allow_empty = True,
            doc = """\
List of Python packages to include in the distribution.
Sub-packages are automatically included.
""",
        ),
    },
    path_inside_wheel = _path_inside_wheel,
)

py_package = rule(
    implementation = py_package_lib.implementation,
    doc = """\
A rule to select all files in transitive dependencies of deps which
belong to given set of Python packages.

This rule is intended to be used as data dependency to py_wheel rule.
""",
    attrs = py_package_lib.attrs,
)

def py_wheel(compatible_with_snowpark = True, **attrs):
    """Modified version of py_wheel rule from rules_python.

    Args:
      compatible_with_snowpark: adds a tag to the wheel to indicate that this
        wheel is compatible with the snowpark running environment.
      **attrs: attributes supported by the native py_wheel rules.
    """

    if compatible_with_snowpark:
        tags = attrs.setdefault("tags", [])
        tags.append(_COMPATIBLE_WITH_SNOWPARK_TAG)
    native_py_wheel(**attrs)

def snowml_wheel(
        name,
        requires,
        extra_requires,
        version,
        deps,
        description_file = None,
        development_status = "Alpha",
        compatible_with_snowpark = True):
    """A SnowML customized wheel definition with lots of default values filled in.

    Args:
      name: Name of the target
      requires: List of required dependencies
      extra_requires(Dict[str, List[str]]): Dict of soft dependencies
      version: Version string
      deps: List of dependencies of type py_package
      development_status: String with PrPr, PuPr & GA
      description_file: Label of readme file.
      compatible_with_snowpark: adds a tag to the wheel to indicate that this
        wheel is compatible with the snowpark running environment.
    """
    dev_status = "Development Status :: 5 - Production/Stable"
    if development_status.lower() == "prpr":
        dev_status = "Development Status :: 3 - Alpha"
    elif development_status.lower() == "pupr":
        dev_status = "Development Status :: 3 - Beta"
    homepage = "https://github.com/snowflakedb/snowflake-ml-python"
    py_wheel(
        name = name,
        author = "Snowflake, Inc",
        author_email = "support@snowflake.com",
        classifiers = [dev_status] +
                      [
                          "Environment :: Console",
                          "Environment :: Other Environment",
                          "Intended Audience :: Developers",
                          "Intended Audience :: Education",
                          "Intended Audience :: Information Technology",
                          "Intended Audience :: System Administrators",
                          "License :: OSI Approved :: Apache Software License",
                          "Operating System :: OS Independent",
                          "Programming Language :: Python :: 3.8",
                          "Programming Language :: Python :: 3.9",
                          "Programming Language :: Python :: 3.10",
                          "Topic :: Database",
                          "Topic :: Software Development",
                          "Topic :: Software Development :: Libraries",
                          "Topic :: Software Development :: Libraries :: Application Frameworks",
                          "Topic :: Software Development :: Libraries :: Python Modules",
                          "Topic :: Scientific/Engineering :: Information Analysis",
                      ],
        description_file = description_file,
        description_content_type = "text/markdown",
        compatible_with_snowpark = compatible_with_snowpark,
        distribution = "snowflake-ml-python",
        extra_requires = extra_requires,
        homepage = homepage,
        project_urls = {
            "Changelog": homepage + "/blob/main/CHANGELOG.md",
            "Documentation": "https://docs.snowflake.com/developer-guide/snowpark-ml",
            "Issues": homepage + "/issues",
            "Source": homepage,
        },
        license = "Apache License, Version 2.0",
        python_requires = ">=3.8,<4",
        python_tag = "py3",
        requires = requires,
        summary = "The machine learning client library that is used for interacting with Snowflake to build machine learning solutions.",
        version = version,
        deps = deps,
    )
