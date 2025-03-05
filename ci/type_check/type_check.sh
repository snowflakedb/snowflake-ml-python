#!/bin/bash

# Usage
# type_check.sh [-a] [-b <bazel_path>]
#
# Flags
#   -a: check all targets (excluding the exempted ones).
#   -b: specify path to bazel.
#
# Inputs
#   - ci/skip_type_checking_targets : a list of target patterns against which
#     typechecking should be enforced.
#
# Action
#   - Create a mypy_test targets to type check all affected targets
# Exit code:
#   0 if succeeds. No target to check means success.
#   1 if there is an error in parsing commandline flag.
#   Otherwise exits with bazel's exit code.
#
# NOTE:
# 1. Ignores all targets that depends on targets in `skip_type_checking_targets`.
# 2. Affected targets also include raw python files on top of bazel build targets whereas ignored_targets don't. Hence
#    we used `kind('py_.* rule')` filter.

set -o pipefail
set -u
set -e

bazel="bazel"
affected_targets=""
PROG=$0

SCRIPT=$(readlink -f "$PROG")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-a] [-b <bazel_path>]"
    exit "${exit_code}"
}

while getopts "ab:h" opt; do
    case "${opt}" in
    a)
        affected_targets="//..."
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

echo "Using bazel: " "${bazel}"
working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT
trap 'rm -rf "${SCRIPTPATH}/runner/BUILD.bazel"' EXIT

if [[ -z "${affected_targets}" ]]; then
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    affected_targets="$(<"${affected_targets_file}")"
fi

printf \
    "let skip_type_checking_targets = set(%s) + set(%s) in \
        let affected_targets = kind('py_.* rule', set(%s)) in \
                let rdeps_targets = rdeps(//..., \$skip_type_checking_targets) in \
                    \$affected_targets except \$rdeps_targets" \
    "$(<"${SCRIPTPATH}/../targets/untyped.txt")" "$(<"${SCRIPTPATH}/../targets/local_only.txt")" "${affected_targets}" >"${working_dir}/type_checked_targets_query"
type_check_targets=$("${bazel}" query --query_file="${working_dir}/type_checked_targets_query" | awk 'NF { print "\""$0"\","}')

echo "${type_check_targets}"

if [[ -z "${type_check_targets}" ]]; then
    echo "No target to do the type checking. Bye!"
    exit 0
fi

cat >"${SCRIPTPATH}/runner/BUILD.bazel" <<EndOfMessage
load("@rules_mypy//:mypy.bzl", "mypy_test")

mypy_test(
    name = "mypy_type_checking",
    deps = [${type_check_targets}],
)
EndOfMessage

set +e
"${bazel}" test \
    --config=all \
    --cache_test_results=no \
    --test_output=errors \
    //ci/type_check/runner:mypy_type_checking
bazel_exit_code=$?

if [[ $bazel_exit_code -eq 0 || $bazel_exit_code -eq 4 ]]; then
    exit 0
fi
exit ${bazel_exit_code}
