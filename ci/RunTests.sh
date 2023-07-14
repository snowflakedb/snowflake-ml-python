#!/bin/bash
# DESCRIPTION: Shell script to run all tests for snowml repository
# AUTHOR     : Tyler Hoyt
# CONTACT    : tyler.hoyt@snowflake.com
# CLI TOOLS  : bazel
#
# Usage
# RunTests.sh [-b <bazel_path>] [-m diff-only|standard]
#
# Flags
#   -b: specify path to bazel.
#   -m: specify the mode from the following options
#       diff-only: run affected tests only. (For merge gate)
#       standard (default): run all tests except auto-generated tests. (For nightly run.)
#

set -o pipefail
set -u

bazel="bazel"
mode="standard"
PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-m diff-only|standard]"
    exit "${exit_code}"
}

while getopts "b:m:h" opt; do
    case "${opt}" in
    m)
        mode="${OPTARG}"
        if ! [[ "${mode}" = "diff-only" || "${mode}" = "standard" ]]; then
            help 1
        fi
        ;;
    b)
        bazel="${OPTARG}"
        ;;
    h)
        help 0
        ;;
    :)
        help 1
        ;;
    ?)
        help 1
        ;;
    esac
done

"${bazel}" clean

working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT

case "${mode}" in
diff-only)
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    test_targets=$(${bazel} query "kind('py_test rule', rdeps(//..., set($(<"${affected_targets_file}"))))")
    ;;
standard)
    test_targets="//..."
    ;;
*)
    help 1
    ;;
esac

test_targets_file=${working_dir}/test_targets
printf "%s" "${test_targets}" > "${test_targets_file}"

"${bazel}" test --cache_test_results=no \
    --test_output=errors \
    --test_tag_filters=-autogen \
    --target_pattern_file "${test_targets_file}"
bazel_exit_code=$?
# Bazel exit code
#   0: Success;
#   4: Build Successful but no tests found
# See https://bazel.build/run/scripts#exit-codes
if [[ ${mode} = "diff-only" && ${bazel_exit_code} -eq 4 ]] ; then
  exit 0
fi
exit $bazel_exit_code
