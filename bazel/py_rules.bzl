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
5. Similarly, a boolean attribute "require_gpu" is available in all the wrapped rules.
   The value of this flag affects which platform and toolchain could be running the target as above.
   Only toolchain supporting GPU would be able to run target tagged with "require_gpu=True"

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
load(":repo_paths.bzl", "check_for_experimental_dependencies", "check_for_test_name", "check_for_tests_dependencies")

def py_genrule(**attrs):
    original_cmd = attrs["cmd"]
    attrs["cmd"] = select({
        "@bazel_tools//src/conditions:windows": "CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1 " + original_cmd,
        "//conditions:default": original_cmd,
    })
    native.genrule(**attrs)

_COMPATIBLE_WITH_SNOWPARK_TAG = "wheel_compatible_with_snowpark"

def _add_target_compatibility_labels(compatible_with_snowpark, require_gpu, attrs):
    if compatible_with_snowpark and require_gpu:
        fail("`require_gpu` is not compatible with snowpark.!")
    if compatible_with_snowpark:
        attrs["target_compatible_with"] = select({
            "//bazel/platforms:extended_conda_channels": [],
            "//bazel/platforms:snowflake_conda_channel": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }) + select({
            "//bazel/platforms:has_gpu": [],
            "//bazel/platforms:no_gpu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif require_gpu:
        attrs["target_compatible_with"] = select({
            "//bazel/platforms:extended_conda_channels": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }) + select({
            "//bazel/platforms:has_gpu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })
    else:
        attrs["target_compatible_with"] = select({
            "//bazel/platforms:extended_conda_channels": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }) + select({
            "//bazel/platforms:has_gpu": [],
            "//bazel/platforms:no_gpu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        })

def py_binary(compatible_with_snowpark = True, require_gpu = False, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_binary](https://docs.bazel.build/versions/master/be/python.html#py_binary) documentation.

    Args:
      compatible_with_snowpark: see file-level document.
      require_gpu: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, require_gpu, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    attrs["env"] = select({
        "@bazel_tools//src/conditions:windows": {"CONDA_DLL_SEARCH_MODIFICATION_ENABLE": "1"},
        "//conditions:default": {},
    })
    native_py_binary(**attrs)

def py_library(compatible_with_snowpark = True, require_gpu = False, **attrs):
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
      require_gpu: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, require_gpu, attrs)

    native_py_library(**attrs)

def py_test(compatible_with_snowpark = True, require_gpu = False, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_test](https://docs.bazel.build/versions/master/be/python.html#py_test) documentation.

    Args:
      compatible_with_snowpark: see file-level document.
      require_gpu: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_test_name(native.package_name(), attrs):
        fail("A test target does not have a valid name!")
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(compatible_with_snowpark, require_gpu, attrs)

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

def _py_wheel_impl(ctx):
    runfiles_root = ctx.executable.wheel_builder.path + ".runfiles"
    workspace_name = ctx.workspace_name
    execution_root_relative_path = "%s/%s" % (runfiles_root, workspace_name)
    wheel_output_dir = ctx.actions.declare_directory("dist")
    ctx.actions.run(
        inputs = [ctx.file.pyproject_toml],
        outputs = [wheel_output_dir],
        executable = ctx.executable.wheel_builder,
        arguments = [
            ctx.file.pyproject_toml.path,
            execution_root_relative_path,
            "--wheel",
            "--outdir",
            wheel_output_dir.path,
        ],
        progress_message = "Building Wheel",
        mnemonic = "WheelBuild",
    )

    return [DefaultInfo(files = depset([wheel_output_dir]))]

_py_wheel = rule(
    attrs = {
        "pyproject_toml": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "wheel_builder": attr.label(
            executable = True,
            cfg = "exec",
        ),
    },
    implementation = _py_wheel_impl,
)

def py_wheel(name, deps, data = [], **kwargs):
    wheel_builder_name = name + "_wheel_builder_main"
    py_binary(
        name = wheel_builder_name,
        srcs = ["@//bazel:wheelbuilder.py"],
        visibility = ["//visibility:public"],
        main = "wheelbuilder.py",
        deps = deps,
        data = data,
    )
    _py_wheel(name = name, wheel_builder = wheel_builder_name, **kwargs)
