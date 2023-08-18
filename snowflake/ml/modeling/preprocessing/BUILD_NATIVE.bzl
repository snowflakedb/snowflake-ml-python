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
        name = "binarizer",
        srcs = [
            "binarizer.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "k_bins_discretizer",
        srcs = [
            "k_bins_discretizer.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "label_encoder",
        srcs = [
            "label_encoder.py",
        ],
        deps = [
            ":init",
            ":ordinal_encoder",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal:type_utils",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "max_abs_scaler",
        srcs = [
            "max_abs_scaler.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "min_max_scaler",
        srcs = [
            "min_max_scaler.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "normalizer",
        srcs = [
            "normalizer.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "one_hot_encoder",
        srcs = [
            "one_hot_encoder.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal:type_utils",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/_internal/utils:identifier",
            "//snowflake/ml/model:model_signature",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "ordinal_encoder",
        srcs = [
            "ordinal_encoder.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal:type_utils",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/_internal/utils:identifier",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "robust_scaler",
        srcs = [
            "robust_scaler.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/_internal/exceptions:exceptions",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_library(
        name = "standard_scaler",
        srcs = [
            "standard_scaler.py",
        ],
        deps = [
            ":init",
            "//snowflake/ml/_internal:telemetry",
            "//snowflake/ml/modeling/framework",
        ],
    )

    py_package(
        name = "preprocessing_pkg",
        packages = ["snowflake.ml"],
        deps = [
            ":binarizer",
            ":k_bins_discretizer",
            ":label_encoder",
            ":max_abs_scaler",
            ":min_max_scaler",
            ":normalizer",
            ":one_hot_encoder",
            ":ordinal_encoder",
            ":robust_scaler",
            ":standard_scaler",
        ],
    )
