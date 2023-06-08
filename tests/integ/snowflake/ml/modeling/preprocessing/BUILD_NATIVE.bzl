load("//bazel:py_rules.bzl", "py_test")

def get_build_rules_for_native_impl():
    SHARD_COUNT = 5
    TIMEOUT = "long"  # 900s

    py_test(
        name = "test_binarizer",
        srcs = ["test_binarizer.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:binarizer",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_k_bins_discretizer",
        srcs = ["test_k_bins_discretizer.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:k_bins_discretizer",
            "//snowflake/ml/utils:connection_params",
            "//snowflake/ml/utils:sparse",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_label_encoder",
        srcs = ["test_label_encoder.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:label_encoder",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_max_abs_scaler",
        srcs = ["test_max_abs_scaler.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:max_abs_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_min_max_scaler",
        srcs = ["test_min_max_scaler.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:min_max_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_normalizer",
        srcs = ["test_normalizer.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:normalizer",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_one_hot_encoder",
        srcs = ["test_one_hot_encoder.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/_internal/utils:identifier",
            "//snowflake/ml/modeling/preprocessing:one_hot_encoder",
            "//snowflake/ml/utils:connection_params",
            "//snowflake/ml/utils:sparse",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_ordinal_encoder",
        srcs = ["test_ordinal_encoder.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:ordinal_encoder",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_robust_scaler",
        srcs = ["test_robust_scaler.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:robust_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_standard_scaler",
        srcs = ["test_standard_scaler.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:standard_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "test_drop_input_cols",
        srcs = ["test_drop_input_cols.py"],
        deps = [
            "//snowflake/ml/modeling/impute:simple_imputer",
            "//snowflake/ml/modeling/pipeline:pipeline",
            "//snowflake/ml/modeling/preprocessing:binarizer",
            "//snowflake/ml/modeling/preprocessing:label_encoder",
            "//snowflake/ml/modeling/preprocessing:max_abs_scaler",
            "//snowflake/ml/modeling/preprocessing:min_max_scaler",
            "//snowflake/ml/modeling/preprocessing:normalizer",
            "//snowflake/ml/modeling/preprocessing:one_hot_encoder",
            "//snowflake/ml/modeling/preprocessing:ordinal_encoder",
            "//snowflake/ml/modeling/preprocessing:robust_scaler",
            "//snowflake/ml/modeling/preprocessing:standard_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )
