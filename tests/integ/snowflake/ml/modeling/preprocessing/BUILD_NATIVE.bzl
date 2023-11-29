load("//bazel:py_rules.bzl", "py_test")

def get_build_rules_for_native_impl():
    SHARD_COUNT = 5
    TIMEOUT = "long"  # 900s

    py_test(
        name = "binarizer_test",
        srcs = ["binarizer_test.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:binarizer",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "k_bins_discretizer_test",
        srcs = ["k_bins_discretizer_test.py"],
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
        name = "label_encoder_test",
        srcs = ["label_encoder_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:label_encoder",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "max_abs_scaler_test",
        srcs = ["max_abs_scaler_test.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:max_abs_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "min_max_scaler_test",
        srcs = ["min_max_scaler_test.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:min_max_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "normalizer_test",
        srcs = ["normalizer_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:normalizer",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "one_hot_encoder_test",
        srcs = ["one_hot_encoder_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/_internal/utils:identifier",
            "//snowflake/ml/modeling/preprocessing:one_hot_encoder",
            "//snowflake/ml/utils:connection_params",
            "//snowflake/ml/utils:sparse",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
        data = ["//tests/integ/snowflake/ml/test_data:UCI_BANK_MARKETING_20COLUMNS.csv"],
    )

    py_test(
        name = "ordinal_encoder_test",
        srcs = ["ordinal_encoder_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:ordinal_encoder",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "robust_scaler_test",
        srcs = ["robust_scaler_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/preprocessing:robust_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "standard_scaler_test",
        srcs = ["standard_scaler_test.py"],
        deps = [
            "//snowflake/ml/modeling/preprocessing:standard_scaler",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )

    py_test(
        name = "drop_input_cols_test",
        srcs = ["drop_input_cols_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
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
