load("//bazel:py_rules.bzl", "py_test")

def get_build_rules_for_native_impl():
    SHARD_COUNT = 5
    TIMEOUT = "long"  # 900s

    py_test(
        name = "simple_imputer_test",
        srcs = ["simple_imputer_test.py"],
        shard_count = SHARD_COUNT,
        timeout = TIMEOUT,
        deps = [
            "//snowflake/ml/modeling/impute:simple_imputer",
            "//snowflake/ml/utils:connection_params",
            "//tests/integ/snowflake/ml/modeling/framework:utils",
        ],
    )
