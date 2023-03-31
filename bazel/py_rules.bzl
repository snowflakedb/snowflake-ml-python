"""py_{binary|libary|test} rules for snowml repository.

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
    native_py_libary = "py_library",
    native_py_test = "py_test",
)
load("@rules_python//python:packaging.bzl", native_py_wheel = "py_wheel")
load(":repo_paths.bzl", "check_for_experimental_dependencies", "check_for_tests_dependencies")


_COMPATIBLE_WITH_SNOWPARK_TAG = "wheel_compatible_with_snowpark"


def _add_target_compatiblity_labels(compatible_with_snowpark, attrs):
    if compatible_with_snowpark:
        attrs["target_compatible_with"] = select({
                "//bazel/platforms:snowflake_conda_channel": [],
                "//bazel/platforms:extended_conda_channels": [],
                "//conditions:default": ["@platforms//:incompatible"],
            })
    else:
        attrs["target_compatible_with"] = select({
                "//bazel/platforms:extended_conda_channels": [],
                "//conditions:default": ["@platforms//:incompatible"],
            })

def py_binary(compatible_with_snowpark=True, **attrs):
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
    _add_target_compatiblity_labels(compatible_with_snowpark, attrs)
    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    native_py_binary(**attrs)

def py_library(compatible_with_snowpark=True, **attrs):
    """Modified version of core py_library to add additional imports and check for experimental dependencies.

    See the Bazel core [py_library](https://docs.bazel.build/versions/master/be/python.html#py_library) documentation.

    Additional import is necessary to expose a libray outside of top-level src directory. For example,
    by defining `//src/snowflake/ml/utils:connection_params` as follows:
    ```
    py_libary(
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
    _add_target_compatiblity_labels(compatible_with_snowpark, attrs)

    native_py_libary(**attrs)

def py_test(compatible_with_snowpark=True, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_test](https://docs.bazel.build/versions/master/be/python.html#py_test) documentation.

    Args:
      compatible_with_snowpark: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatiblity_labels(compatible_with_snowpark, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    native_py_test(**attrs)

def py_wheel(compatible_with_snowpark=True, **attrs):
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
