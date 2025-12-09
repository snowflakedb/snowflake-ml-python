"""py_{binary|library|test} rules for snowml repository.

Overriding default implementation of py_{binary|library|test} to add additional features:
1. Codebase is split between 'src' & 'tests' top-level directories. We did not want python import with
   'src.snowflake.ml...'. We just want 'snowflake.ml....'. Hence I added to add additional imports statements
   to `py_library`.
2. Added check so that a target in `src` cannot depend on something in `tests`.
3. Similarly, added a check that a non-experimental target cannot depend on `experimental` target.
4. optional_dependencies is used to add target_compatible_with label to the target. This is used to
   add compatibility with different conda channels. If the target is using an optional dependency
   then this is need to be set to align with extra_requirements_tag in requirements.yml file.

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
load("//bazel/platforms:optional_dependency_groups.bzl", "OPTIONAL_DEPENDENCY_GROUPS")
load(":repo_paths.bzl", "check_for_experimental_dependencies", "check_for_test_name", "check_for_tests_dependencies")

def py_genrule(**attrs):
    original_cmd = attrs["cmd"]
    attrs["cmd"] = select({
        "@bazel_tools//src/conditions:windows": "CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1 " + original_cmd,
        "//conditions:default": original_cmd,
    })
    native.genrule(**attrs)

def _add_target_compatibility_labels(optional_dependencies, attrs):
    compatibility = ["//bazel/platforms:core_conda_channel"]
    if optional_dependencies:
        found_matching_group = False
        for group_name, group in OPTIONAL_DEPENDENCY_GROUPS.items():
            dependency_in_group = True
            for dependency in optional_dependencies:
                if dependency not in group:
                    dependency_in_group = False
                    break
            if dependency_in_group:
                compatibility.append("//bazel/platforms:{}_conda_channel".format(group_name))
                found_matching_group = True
                break

        if not found_matching_group:
            fail(
                "Could not find a compatible dependency group containing all of: {}".format(
                    ", ".join(optional_dependencies),
                ),
            )
    attrs["target_compatible_with"] = compatibility

def py_binary(optional_dependencies = None, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_binary](https://docs.bazel.build/versions/master/be/python.html#py_binary) documentation.

    Args:
      optional_dependencies: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(optional_dependencies, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    attrs["env"] = select({
        "@bazel_tools//src/conditions:windows": {"CONDA_DLL_SEARCH_MODIFICATION_ENABLE": "1"},
        "//conditions:default": {},
    })
    native_py_binary(**attrs)

def py_library(**attrs):
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
      optional_dependencies: see file-level document.
      **attrs: Rule attributes
    """
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")

    native_py_library(**attrs)

def py_test(optional_dependencies = None, **attrs):
    """Modified version of core py_binary to add check for experimental dependencies.

    See the Bazel core [py_test](https://docs.bazel.build/versions/master/be/python.html#py_test) documentation.

    Args:
      optional_dependencies: see file-level document.
      **attrs: Rule attributes
    """

    # Validate required feature area tag
    _validate_feature_area_tag(attrs)

    if not check_for_test_name(native.package_name(), attrs):
        fail("A test target does not have a valid name!")
    if not check_for_tests_dependencies(native.package_name(), attrs):
        fail("A target in src cannot depend on packages in tests!")
    if not check_for_experimental_dependencies(native.package_name(), attrs):
        fail("Non Experimental Target cannot depend on experimental library!")
    _add_target_compatibility_labels(optional_dependencies, attrs)

    # Disable bazel's behavior to add __init__.py files to modules by default. This causes import errors. Context:
    # * https://bazel.build/reference/be/python#py_test.legacy_create_init
    # * https://github.com/bazelbuild/rules_python/issues/55
    attrs["legacy_create_init"] = 0
    native_py_test(**attrs)

def _validate_feature_area_tag(attrs):
    """Validates that a py_test target has a required feature area tag.

    Args:
        attrs: Rule attributes
    """
    VALID_FEATURE_AREAS = ["model_registry", "feature_store", "jobs", "observability", "experiment_tracking", "cortex", "core", "modeling", "model_serving", "data", "none"]

    tags = attrs.get("tags", [])
    feature_tags = [tag for tag in tags if tag.startswith("feature:")]

    if not feature_tags:
        fail("py_test target '{}' must have a feature area tag in format 'feature:<feature_area>'. Valid feature areas: {}. For help: bazel run //bazel:check_feature_tags".format(
            attrs.get("name", "unknown"),
            ", ".join(VALID_FEATURE_AREAS),
        ))

    if len(feature_tags) > 1:
        fail("py_test target '{}' can only have one feature area tag, found: {}".format(
            attrs.get("name", "unknown"),
            ", ".join(feature_tags),
        ))

    feature_area = feature_tags[0][8:]  # Remove "feature:" prefix
    if feature_area not in VALID_FEATURE_AREAS:
        fail("py_test target '{}' has invalid feature area '{}'. Valid feature areas: {}".format(
            attrs.get("name", "unknown"),
            feature_area,
            ", ".join(VALID_FEATURE_AREAS),
        ))

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
            "--sdist",
            "--outdir",
            wheel_output_dir.path,
        ],
        use_default_shell_env = True,
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
