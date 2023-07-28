load("//bazel:py_rules.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package")

def get_build_rules_for_native_impl():
    py_library(
        name = "init",
        srcs = [
            "__init__.py",
        ],
        deps = [
            "//snowflake/ml/_internal:init_utils",
        ],
    )

    py_library(
        name = "simple_imputer",
        srcs = [
            "simple_imputer.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_package(
        name = "impute_pkg",
        packages = ["snowflake.ml"],
        deps = [
            ":simple_imputer",
        ],
    )
