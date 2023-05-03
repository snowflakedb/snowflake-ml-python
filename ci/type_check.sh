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
#   - /tmp/affected_targets/targets : a list of targets affected by the change.
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
# 1. Ignores all targets that depends on (1) targets with tag "skip_mypy_check" (2) targets in `type_ignored_targets`.
# 2. Affected targets also include raw python files on top of bazel build targets whereas ignored_targets don't. Hence
#    we used `kind('py_.* rule')` filter.

set -o pipefail
set -e

bazel="bazel"
affected_targets=""

while getopts "ab:" opt; do
  case "${opt}" in
    a)
        affected_targets="//..."
        ;;
    b)
        bazel="${OPTARG}"
        ;;
    :)
        echo "Option --bazel requires an argument."
        exit 1
        ;;
    ?)
        echo "Invalid option."
        echo "Usage: $0 [-a] [-b <bazel_path>]"
        exit 1
        ;;
  esac
done

echo "Using bazel: " "${bazel}"

if [[ -z "${affected_targets}" ]]; then
    affected_targets="$(</tmp/affected_targets/targets)"
fi

printf \
    "let type_ignored_targets = set(%s) in \
        let affected_targets = kind('py_.* rule', set(%s)) in \
            let skipped_targets = attr('tags', '[\[ ]skip_mypy_check[,\]]', \$affected_targets) in \
                let rdeps_targets = rdeps(//..., \$type_ignored_targets) union rdeps(//..., \$skipped_targets) in \
                    \$affected_targets except \$rdeps_targets" \
    "$(<ci/type_ignored_targets)" "${affected_targets}" > /tmp/type_checked_targets_query
"${bazel}" query --query_file=/tmp/type_checked_targets_query > /tmp/type_checked_targets
echo "Type checking the following targets:" "$(</tmp/type_checked_targets)"
set +e
"${bazel}" build \
    --keep_going \
    --config=typecheck \
    --color=yes \
    --target_pattern_file=/tmp/type_checked_targets
bazel_exit_code=$?
if [ $bazel_exit_code -ne 0 ] || [ $bazel_exit_code -ne 4 ]; then
    exit $bazel_exit_code
fi
exit 0
