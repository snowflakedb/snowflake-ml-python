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

set -o pipefail
set -u
set -e

bazel="bazel"
mode="continuous_run"
PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-m merge_gate|continuous_run]"
    exit "${exit_code}"
}

while getopts "b:m:h" opt; do
    case "${opt}" in
    m)
        if [[ "${OPTARG}" = "merge_gate" || "${OPTARG}" = "continuous_run" ]] ; then
            mode="${OPTARG}"
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

tag_filter="--test_tag_filters="

case "${mode}" in
merge_gate)
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    tag_filter="--test_tag_filters=-autogen,-skip_continuous_test"

    # Notice that we should include all kinds of test here.
    test_targets=$(${bazel} query "kind('.*_test rule', rdeps(//... - //snowflake/ml/experimental/... - set($(<ci/skip_merge_gate_targets)), set($(<"${affected_targets_file}"))))")
    ;;
continuous_run)
    test_targets=$(${bazel} query "kind('.*_test rule', //... - //snowflake/ml/experimental/...)")
    ;;
*)
    help 1
    ;;
esac

test_targets_file=${working_dir}/test_targets
printf "%s" "${test_targets}" >"${test_targets_file}"

set +e
"${bazel}" test --cache_test_results=no \
    --test_output=errors \
    "${tag_filter}" \
    --target_pattern_file "${test_targets_file}"
bazel_exit_code=$?
# Bazel exit code
#   0: Success;
#   4: Build Successful but no tests found
# See https://bazel.build/run/scripts#exit-codes
if [[ ${mode} = "merge_gate" && ${bazel_exit_code} -eq 4 ]]; then
    exit 0
fi
exit $bazel_exit_code
