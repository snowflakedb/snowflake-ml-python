#!/bin/bash

# Usage
# type_check.sh [-a] [-b <bazel_path>]
#
# Flags
#   -a: check all targets (excluding the exempted ones).
#   -b: specify path to bazel.
#
# Inputs
#   - ci/type_ignored_targets : a list of target patterns against which
#     typechecking should be enforced. Not required if "-a" is specified.
#
# Action
#   - Performs typechecking against the intersection of
#     type checked targets and affected targets.
# Exit code:
#   0 if succeeds. No target to check means success.
#   1 if there is an error in parsing commandline flag.
#   Otherwise exits with bazel's exit code.
#
# NOTE:
# 1. Ignores all targets that depends on targets in `type_ignored_targets`.
# 2. Affected targets also include raw python files on top of bazel build targets whereas ignored_targets don't. Hence
#    we used `kind('py_.* rule')` filter.

set -o pipefail
set -u
set -e

bazel="bazel"
affected_targets=""
PROG=$0

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

if [[ -z "${affected_targets}" ]]; then
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    affected_targets="$(<"${affected_targets_file}")"
fi

printf \
    "let type_ignored_targets = set(%s) in \
        let affected_targets = kind('py_.* rule', set(%s)) in \
                let rdeps_targets = rdeps(//..., \$type_ignored_targets) in \
                    \$affected_targets except \$rdeps_targets" \
    "$(<ci/type_ignored_targets)" "${affected_targets}" >"${working_dir}/type_checked_targets_query"
"${bazel}" query --query_file="${working_dir}/type_checked_targets_query" >"${working_dir}/type_checked_targets"
echo "Type checking the following targets:" "$(<"${working_dir}/type_checked_targets")"

set +e
"${bazel}" build \
    --keep_going \
    --config=typecheck \
    --color=yes \
    --target_pattern_file="${working_dir}/type_checked_targets"
bazel_exit_code=$?

if [[ $bazel_exit_code -eq 0 || $bazel_exit_code -eq 4 ]]; then
    exit 0
fi
exit ${bazel_exit_code}
