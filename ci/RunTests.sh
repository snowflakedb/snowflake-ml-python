#!/bin/bash
# DESCRIPTION: Shell script to run all tests for snowml repository
# AUTHOR     : Tyler Hoyt
# CONTACT    : tyler.hoyt@snowflake.com
# CLI TOOLS  : bazel
#
# Usage
# RunTests.sh [-b <bazel_path>] [-m merge_gate|continuous_run]
#
# Flags
#   -b: specify path to bazel.
#   -m: specify the mode from the following options
#       merge_gate: run affected tests only.
#       continuous_run (default): run all tests except auto-generated tests and tests with
#           'skip_continuous_test' filter. (For nightly run.)
#

./ci/RunBazelAction.sh test "$@"
