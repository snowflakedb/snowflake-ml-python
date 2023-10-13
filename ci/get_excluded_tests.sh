#!/bin/bash

# Usage
# exclude_tests.sh [-b <bazel_path>] [-f <output_path>] [-m unused|unaffected|all]
#
# Flags
# -b: specify path to bazel
# -f: specify output file path
# -m: specify the mode from the following options
#       unused: exclude integration tests whose dependency is not part of the wheel package.
#               The missing dependency could happen when a new operator is being developed,
#               but not yet released.
#       unaffected: exclude integration tests whose dependency is not part of the affected targets
#                   compare to the the merge base to main of current revision.
#       all (default): exclude the union of above rules.
#

set -o pipefail
set -u

PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} [-b <bazel_path>] [-f <output_path>] [-m unused|unaffected|all]"
    exit "${exit_code}"
}

echo "Running ${PROG}"

bazel="bazel"
output_path="/tmp/files_to_exclude"
mode="all"

while getopts "b:f:m:h" opt; do
    case "${opt}" in
    b)
        bazel=${OPTARG}
        ;;
    f)
        output_path=${OPTARG}
        ;;
    m)
        mode=${OPTARG}
        if ! [[ $mode = "unused" || $mode = "unaffected" || $mode = "all" ]]; then
            help 1
        fi
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

working_dir=$(mktemp -d "/tmp/tmp_XXXXX")
trap 'rm -rf "${working_dir}"' EXIT

if [[ $mode = "unused" || $mode = "all" ]]; then
    # Compute missing dependencies by subtracting deps included in wheel from deps required by tests.
    # We only care about dependencies in //snowflake/ml since that's our dev directory.
    # Reverse search on testing files depending on missing deps and exclude those.

    unused_test_rule_file=${working_dir}/unused_test_rule

    # -- Begin of Query Rules Heredoc --
    cat >"${unused_test_rule_file}" <<EndOfMessage
    let missing_deps = filter('//snowflake/ml[:/].*', kind('py_library rule', deps(tests/...) except deps(snowflake/ml:wheel))) in
        kind('source file', labels(srcs, kind('py_test rule', rdeps(//tests/..., \$missing_deps, 1))))
EndOfMessage
    # -- End of Query Rules Heredoc --

    unused_test_targets=$("${bazel}" query --query_file="${unused_test_rule_file}")
fi

if [[ $mode = "unaffected" || $mode = "all" ]]; then
    affected_targets_file="${working_dir}/affected_targets"
    ./bazel/get_affected_targets.sh -b "${bazel}" -f "${affected_targets_file}"

    unaffected_test_rule_file=${working_dir}/unaffected_test_rule

    # -- Begin of Query Rules Heredoc --
    cat >"${unaffected_test_rule_file}" <<EndOfMessage
    let unaffected_targets = //tests/... - rdeps(//tests/..., set($(<"${affected_targets_file}"))) in
        kind('source file', labels(srcs, set($(<ci/skip_merge_gate_targets)) + kind('py_test rule', \$unaffected_targets)) - labels(srcs, rdeps(//tests/..., set($(<"${affected_targets_file}")))))
EndOfMessage
    # -- End of Query Rules Heredoc --

    unaffected_test_targets=$("${bazel}" query --query_file="${unaffected_test_rule_file}")
fi

targets_to_exclude_file="${working_dir}/targets_to_exclude_file"

case "${mode}" in
unused)
    echo "${unused_test_targets}" >"${targets_to_exclude_file}"
    ;;
unaffected)
    echo "${unaffected_test_targets}" >"${targets_to_exclude_file}"
    ;;
all)
    # Concat and deduplicate.
    targets_to_exclude=$(printf "%s\n%s\n" "${unused_test_targets}" "${unaffected_test_targets}" | awk '!a[$0]++')
    echo "${targets_to_exclude}" >"${targets_to_exclude_file}"
    ;;
*)
    help 1
    ;;
esac

excluded_test_source_rule_file=${working_dir}/excluded_test_source_rule

printf "kind('source file', set(%s))" "$(<"${targets_to_exclude_file}")" >"${excluded_test_source_rule_file}"

${bazel} query --query_file="${excluded_test_source_rule_file}" \
    --output location |
    grep -o "source file.*" |
    awk -F// '{print $2}' |
    sed -e 's/:/\//g' >"${output_path}"

echo "Tests getting excluded:"

cat "${output_path}"

echo "Done running ${PROG}"
