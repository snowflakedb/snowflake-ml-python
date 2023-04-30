#!/usr/bin/env python3
""" Main tool for auto-generating wrapped sklearn transforms.

    Example usage:

    cd ~/src/snowml/
    python3 codegen/estimator_autogen_tool.py \
        --template_file=codegen/sklearn_wrapper_template.py_template \
        --output_path="" \
        --module="sklearn.linear_model" \
        --gen_mode='SRC'
"""
import os
from typing import List

from absl import app, flags, logging
from sklearn_wrapper_autogen import AutogenTool, GenMode

FLAGS = flags.FLAGS
flags.DEFINE_string("template_file", None, "Path to the file containing the {transformer|test} template.")
flags.DEFINE_string(
    "output_path",
    "",
    "Directory where to write the generated transformers (or tests). "
    + "Existing files will be overwritten. Directory path will be created if missing.",
)
flags.DEFINE_string(
    "module",
    "sklearn.linear_model",
    "Scikit-learn modules to run autogeneration for.",
)
flags.DEFINE_list(
    "class_list",
    None,
    "List of classes to generate transformers for. If not set, "
    + "transformers will be generated for all relevant classes in the module.",
)
flags.DEFINE_string(
    "gen_mode",
    None,
    "Options: ['SRC', 'TEST']."
    + "SRC mode generates source code for snowflake wrapper for all the estimator objects in the given modules.\n"
    + "TEST mode generates integration tests for all the auto gnerated python wrappers in the given module.\n",
)
flags.DEFINE_string(
    "bazel_out_dir", None, "Takes bazel out directory as input to compute relative path to bazel-bin folder"
)

# Required flag.
flags.mark_flag_as_required("gen_mode")
flags.mark_flag_as_required("template_file")
flags.mark_flag_as_required("output_path")


def main(argv: List[str]) -> None:
    del argv  # Unused.

    gen_mode = None
    for member in GenMode:
        if member.name == FLAGS.gen_mode:
            gen_mode = member

    if not gen_mode:
        raise AssertionError(
            f"Unexpected value for gen_mode flag : {FLAGS.gen_mode}. " "Expected values are {'SRC', 'TEST'}"
        )

    actual_output_path = FLAGS.output_path

    # Comput relative path of bazel-bin folder.
    if FLAGS.bazel_out_dir:
        expected_suffix = AutogenTool.module_root_dir(module_name=FLAGS.module)
        expected_suffix = os.path.normpath(os.path.join(actual_output_path, expected_suffix))

        bazel_out_dir = FLAGS.bazel_out_dir
        if not bazel_out_dir.endswith(expected_suffix):
            raise AssertionError(
                f"genrule output dir $(RULEDIR) {bazel_out_dir} is expected to end with suffix {expected_suffix}"
            )
        relative_bazel_bin_path = os.path.normpath(remove_suffix(bazel_out_dir, expected_suffix))
        logging.info(f"Relative bazel bin path : {relative_bazel_bin_path}")

        # Update src and test output paths by prepending relative bazel-bin path.
        actual_output_path = os.path.normpath(os.path.join(relative_bazel_bin_path, actual_output_path))

    autogen_tool = AutogenTool(
        gen_mode=gen_mode,
        template_path=FLAGS.template_file,
        class_list=FLAGS.class_list,
        output_path=actual_output_path,
    )
    autogen_tool.generate(FLAGS.module)


def remove_suffix(input_string: str, suffix: str) -> str:
    """Remove suffix from input string.

    Args:
        input_string: Source string.
        suffix: Suffix to be removed from source string.

    Returns:
        Result of removing `suffix` from `input_string`.
    """
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


if __name__ == "__main__":
    app.run(main)
