"""autogen_{init_file_for_module|estimators|tests_for_estimators} rules for snowml repository.

Helper functions to autogenerate genrules and build rules for the following
1. Snowflake wrappers for estimators from a given modue.
2. Integration tests for wrapper classes.
3. Init file for the auto-generated wrappers module.
"""

load("@rules_python//python:packaging.bzl", native_py_package = "py_package")
load("//bazel:py_rules.bzl", "py_library", "py_test")

AUTO_GEN_TOOL_BAZEL_PATH = "//codegen:estimator_autogen_tool"
ESTIMATOR_TEMPLATE_BAZEL_PATH = "//codegen:sklearn_wrapper_template.py_template"
ESTIMATOR_TEST_TEMPLATE_BAZEL_PATH = (
    "//codegen:transformer_autogen_test_template.py_template"
)
INIT_TEMPLATE_BAZEL_PATH = "//codegen:init_template.py_template"
SRC_OUTPUT_PATH = ""
TEST_OUTPUT_PATH = "tests/integ"

def autogen_init_file_for_module(module):
    """Generates `genrule` and `py_library` rules for init file.

    List of generated build rules
        1. `genrule` with label 'generate_init_file' to auto-generate init file for the given module.
        2. `py_library` rule with label 'init' to build the output of `generate_inti_file`

    Args:
        module (str) : Name of the module to auto-generate init file for.
    """

    native.genrule(
        name = "generate_init_file",
        outs = ["__init__.py"],
        tools = [AUTO_GEN_TOOL_BAZEL_PATH],
        srcs = [INIT_TEMPLATE_BAZEL_PATH],
        cmd = "cat $(location {}) > $@".format(INIT_TEMPLATE_BAZEL_PATH),
    )

    py_library(
        name = "init",
        srcs = [":generate_init_file"],
        deps = ["//snowflake/ml/_internal:init_utils"],
        tags = ["skip_mypy_check"],
    )

def get_genrule_cmd(gen_mode, template_path, module, output_path):
    return """$(location {}) \\
                --bazel_out_dir=$(RULEDIR) \\
                --gen_mode={} \\
                --template_file=$(location {}) \\
                --module={} \\
                --output_path={} \\
                --class_list={{}}""".format(AUTO_GEN_TOOL_BAZEL_PATH, gen_mode, template_path, module, output_path)

def autogen_estimators(module, estimator_info_list):
    """ Generates `genrule` and `py_library` rules for every estimator in the estimator_info_list
    List of generated build rules for every class in the estimator_info_list
        1. `genrule` with label `generate_<estimator-class-name-snakecase>` to auto-generate
            snowflake wrapper for the estimator.
        2. `py_library` rule with label `<estimator-class-name-snakecase>` to build the auto-generated
            python files from the  `generate_<estimator-class-name-snakecase>` rule.
    """
    cmd = get_genrule_cmd(
        gen_mode = "SRC",
        template_path = ESTIMATOR_TEMPLATE_BAZEL_PATH,
        module = module,
        output_path = SRC_OUTPUT_PATH,
    )

    for e in estimator_info_list:
        native.genrule(
            name = "generate_{}".format(e.normalized_class_name),
            outs = ["{}.py".format(e.normalized_class_name)],
            tools = [AUTO_GEN_TOOL_BAZEL_PATH],
            srcs = [ESTIMATOR_TEMPLATE_BAZEL_PATH],
            cmd = cmd.format(e.class_name),
        )

        py_library(
            name = "{}".format(e.normalized_class_name),
            srcs = [":generate_{}".format(e.normalized_class_name)],
            deps = [
                ":init",
                "//snowflake/ml/framework:framework",
                "//snowflake/ml/utils:telemetry",
                "//snowflake/ml/utils:temp_file_utils",
                "//snowflake/ml/utils:query_result_checker",
                "//snowflake/ml/utils:pkg_version_utils",
            ],
            tags = ["skip_mypy_check"],
        )

    native_py_package(
        name = "{}_pkg".format(module.lower().replace(".", "_")),
        packages = ["snowflake.ml"],
        deps = [
            ":{}".format(e.normalized_class_name)
            for e in estimator_info_list
        ],
        tags = ["skip_mypy_check"],
    )

def autogen_tests_for_estimators(module, module_root_dir, estimator_info_list):
    """Generates `genrules` and `py_test` rules for every estimator in the estimator_info_list
     List of generated build rules for every class in the estimator_info_list
        1. `genrule` with label `generate_test_<estimator-class-name-snakecase>` to auto-generate
            integration test for the estimator's wrapper class.
        2. `py_test` rule with label `test_<estimator-class-name-snakecase>` to build the auto-generated
            test files from the  `generate_test_<estimator-class-name-snakecase>` rule.
    """
    cmd = get_genrule_cmd(
        gen_mode = "TEST",
        template_path = ESTIMATOR_TEST_TEMPLATE_BAZEL_PATH,
        module = module,
        output_path = TEST_OUTPUT_PATH,
    )

    for e in estimator_info_list:
        native.genrule(
            name = "generate_test_{}".format(e.normalized_class_name),
            outs = ["test_{}.py".format(e.normalized_class_name)],
            tools = [AUTO_GEN_TOOL_BAZEL_PATH],
            srcs = [ESTIMATOR_TEST_TEMPLATE_BAZEL_PATH],
            cmd = cmd.format(e.class_name),
        )

        py_test(
            name = "test_{}".format(e.normalized_class_name),
            srcs = [":generate_test_{}".format(e.normalized_class_name)],
            deps = [
                "//{}:{}".format(module_root_dir, e.normalized_class_name),
                "//snowflake/ml/utils:connection_params",
            ],
            legacy_create_init = 0,
            shard_count = 5,
            timeout = "moderate",
            tags = ["autogen", "skip_mypy_check"],
        )
